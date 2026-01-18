from collections import OrderedDict
import copy
import io
from itertools import chain
import logging
from math import inf
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.autograd import profiler
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import SGD, Optimizer
from fairscale.internal.params import calc_grad_norm, get_global_rank, recursive_copy_to_device
from fairscale.nn.misc import ParamBucket
class OSS(Optimizer):
    """Wraps an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>`
    optimizer and shards its state as described by ZeRO_.
    ::

        opt = OSS(params, optim=torch.optim.Adam, lr=0.01)

    .. _ZeRO: https://arxiv.org/abs/1910.02054

    We use a greedy algorithm to pack a number of parameters
    at each rank. Each parameter belongs to a single rank and
    is not divided among rank.

    After each rank completed their parameter update, they broadcast
    the new version of the parameters to all other ranks to synchronize
    the parameters for next round forward/backward computation.

    Args:
        params (list of tensors):
            parameters to be optimized
    Keyword Args:
        optim (torch.nn.Optimizer):
            optimizer to shard (default: SGD)
        group (group):
            torch.distributed group (default: group.WORLD)
        broadcast_buffer_size (int):
            (deprecated) used to cap the size of the broadcast buffers, not being used anymore.
        broadcast_fp16 (bool):
            Compress the model shards in fp16 before sharing them in between ranks.
            This is safe to use when PyTorch AMP is activated. Without torch AMP this will lead to a slight
            degradation in terms of accuracy.
        force_broadcast_object (bool):
            If True, '_broadcast_object' will be used for rebuilding the sharded optimizer.
            If False, whether to use '_broadcast_object' or 'dist.broadcast_object_list' will be determined by GPU capabilities.
            This feature is needed since some newer GPUs still get some memory issues when applying dist.broadcast_object_list.

    .. warning: the communication patterns that OSS use depend on the "trainability" graph,
        meaning that all the parameters which `require_grad` are handled differently. This is
        not reevaluated at every step, please use `refresh_trainable()` if your model changed
        (freeze or unfreeze for instance).
        If used with :class:<fairscale.nn.ShardedDDP> then an automatic change detection is possible,
        via the `auto_refresh_trainable` parameter.
    """
    optim: Optimizer
    in_super_constructor: bool

    def __init__(self, params: _params_t, optim: Type[Optimizer]=SGD, group: Optional[Any]=None, broadcast_buffer_size: int=-1, broadcast_fp16: bool=False, force_broadcast_object: bool=False, **default: Any):
        self.in_super_constructor = True
        super().__init__(params, default)
        self.in_super_constructor = False
        self.__per_device_params: Dict[torch.device, List[List[Parameter]]] = OrderedDict()
        self.__param_rank: Dict[torch.Tensor, int] = {}
        self._partition_parameters: List[List[dict]] = []
        self.__param_to_index: Dict[int, int] = {}
        self.__local_params: Optional[List[torch.Tensor]] = None
        self._optim_defaults = default
        self._optim_constructor = optim
        self.group = group if group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.group)
        self.backend = dist.get_backend(self.group)
        self.rank = dist.get_rank(self.group)
        self.global_rank = get_global_rank(self.group, self.rank)
        self._local_to_global_rank = [get_global_rank(self.group, i) for i in range(self.world_size)]
        self.broadcast_fp16 = broadcast_fp16
        self.force_broadcast_object = force_broadcast_object
        self.buckets: Dict[torch.device, Dict[int, ParamBucket]] = {}
        self._all_states: List[Dict[str, Any]] = []
        self._default_device = torch.device('cpu')
        self.refresh_trainable()

    def partition_parameters(self) -> List[List[dict]]:
        """Partitions parameters across distributed data parallel ranks.

        Returns a list of param_groups (which is a list of dict) where each
        element of the list contains the param_groups for a rank. Element 0
        corresponds to rank 0, etc. We need all the ranks for the broadcast
        inside step().
        """
        if len(self._partition_parameters) == 0:
            self._partition_parameters = [list() for _ in range(self.world_size)]
            sizes = [0] * self.world_size
            for param_group in self.param_groups:
                param_lists: List[List] = [list() for _ in range(self.world_size)]
                for param in param_group['params']:
                    rank = sizes.index(min(sizes))
                    param_lists[rank].append(param)
                    if param.requires_grad:
                        sizes[rank] += param.numel()
                    else:
                        sizes[rank] += 1
                for rank, params in enumerate(param_lists):
                    param_group_rank = copy.copy(param_group)
                    param_group_rank['params'] = params
                    self._partition_parameters[rank].append(param_group_rank)
        return self._partition_parameters

    def step(self, closure: Optional[Callable[[], float]]=None, **kwargs: Any) -> Optional[float]:
        """Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note: Any extra parameter is passed to the base optimizer as-is"""
        OSS._sync_param_groups(self.param_groups, self.optim.param_groups)
        with profiler.record_function('fairscale::oss::refresh_trainable'):
            if self._default_device.type != self.param_groups[0]['params'][0].device.type:
                logging.info('OSS detected that the parameter changed devices, re-allocating buffers')
                self._clear_cache()
                self.refresh_trainable()
        with profiler.record_function('fairscale::oss::optim_step'):
            if closure is not None:
                loss = self.optim.step(closure=closure, **kwargs)
            else:
                loss = self.optim.step(**kwargs)
        self._broadcast_params()
        OSS._sync_param_groups(self.optim.param_groups, self.param_groups)
        return loss

    def clip_grad_norm(self, max_norm: Union[float, int], norm_type: Union[float, int]=2.0, filter_params_fn: Callable[[Any], Any]=None) -> torch.Tensor:
        """
        Clip all gradients at this point in time. The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Arguments:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        .. note: This is analogous to `torch.nn.utils.clip_grad_norm_` but handles the partitioning and multiple devices per rank
            under the hood. The default torch util is not applicable here, because each rank only has a partial view of all the grads
            in the model, so calling it in the OSS context would lead to different scaling being applied per subset of model parameters

        .. warning: This needs to be called on all ranks, since synchronization primitives will be used

        """
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        with profiler.record_function('fairscale::oss::clip_grad_norm'):
            local_params = filter_params_fn(self._local_params) if filter_params_fn is not None else self._local_params
            local_norm = calc_grad_norm(local_params, norm_type).to(self._default_device)
            if norm_type == inf:
                total_norm = local_norm
                dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=dist.group.WORLD)
            else:
                total_norm = local_norm ** norm_type
                dist.all_reduce(total_norm)
                total_norm = total_norm ** (1.0 / norm_type)
            clip_coef = torch.tensor(max_norm, dtype=total_norm.dtype, device=total_norm.device) / (total_norm + 1e-06)
            if clip_coef < 1:
                for device, device_params in self._per_device_params.items():
                    for p in filter(lambda x: x.grad is not None, device_params[self.rank]):
                        p.grad.detach().mul_(clip_coef.to(device))
        return total_norm

    def consolidate_state_dict(self, recipient_rank: int=0) -> None:
        """Update the consolidated state_dict list, one per rank.

        Arguments:
            recipient_rank (int): on which rank to materialize the full state dict.
            -1 is a special value, which means that all ranks should have the state

        .. warning: This needs to be called on all replicas"""
        OSS._sync_param_groups(self.param_groups, self.optim.param_groups)
        logging.debug('Pulling the sharded optimizer state from all replicas')
        self._all_states = []
        should_collect_state = self.rank == recipient_rank or recipient_rank == -1
        should_send_state = self.rank != recipient_rank
        dist_device = torch.device('cuda') if self.backend == dist.Backend.NCCL else self._default_device
        for rank in range(self.world_size):
            if rank == self.rank:
                if should_collect_state:
                    logging.debug('Saving self state')
                    self._all_states.append(recursive_copy_to_device(self.optim.state_dict(), non_blocking=True, device=torch.device('cpu')))
                state_to_share = self.optim.state_dict() if should_send_state else torch.tensor([0], dtype=torch.uint8, device=dist_device)
                if self.force_broadcast_object or _gpu_capabilities_older_than_50():
                    _broadcast_object(state_to_share, src_rank=self.global_rank, group=self.group, dist_device=dist_device)
                else:
                    obj_list = [state_to_share]
                    dist.broadcast_object_list(obj_list, src=self.global_rank, group=self.group)
            else:
                if self.force_broadcast_object or _gpu_capabilities_older_than_50():
                    replica_state = _broadcast_object(torch.tensor([0], dtype=torch.uint8, device=dist_device), src_rank=self._local_to_global_rank[rank], group=self.group, dist_device=dist_device)
                else:
                    obj_list = [torch.tensor([0], dtype=torch.uint8, device=dist_device)]
                    dist.broadcast_object_list(obj_list, src=self._local_to_global_rank[rank], group=self.group)
                    replica_state = obj_list[0]
                if should_collect_state:
                    self._all_states.append(recursive_copy_to_device(replica_state, non_blocking=True, device=torch.device('cpu')))
                logging.debug('State from rank %s received', rank)

    def state_dict(self, all_ranks: bool=False) -> Dict[str, Any]:
        """Return the last known global optimizer state. The returned state is compatible with Pytorch, in that the
        sharded properties are not exposed.


        Arguments:
            all_ranks (bool): materialize the state on all ranks. In that case, `.state_dict()` needs to be called on
            all ranks

        Returns:
            a dict with two entries
                * state - a dict holding current optimization state. Its content
                    differs between optimizer classes.

                * param_groups - a dict containing all parameter groups

        .. warning:
            Returning the global state is limited to the replica which was responsible for the consolidation,
            if `all_ranks` was not set to `True`. In that case, the state may also not be up to date,
            depending on when `consolidate_state_dict` was last called.
        """
        if not all_ranks and len(self._all_states) == 0:
            raise RuntimeError('Optimizer state has not been consolidated on this rank.                 Please call `consolidate_state_dict()` on all ranks beforehand if you meant to save the global state')
        if all_ranks:
            self.consolidate_state_dict(recipient_rank=-1)
        state_dict = super().state_dict()
        for rank, s in enumerate(self._all_states):
            for local_pg, global_pg in zip(s['param_groups'], self.partition_parameters()[rank]):
                local_index_to_param_id = {i_param: id(global_pg['params'][i]) for i, i_param in enumerate(local_pg['params'])}
                for local_param_index in local_pg['params']:
                    if local_param_index in s['state'].keys():
                        global_id = self._param_to_index[local_index_to_param_id[local_param_index]]
                        state_dict['state'][global_id] = s['state'][local_param_index]
        state_dict['state'] = dict(sorted(state_dict['state'].items()))
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the global parameter groups as well as the shard.

        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`
        """
        id_map = {old_id: p for old_id, p in zip(chain.from_iterable((g['params'] for g in state_dict['param_groups'])), chain.from_iterable((g['params'] for g in self.param_groups)))}
        for key, value in state_dict['state'].items():
            param = id_map[key]
            if self._param_to_rank[param] != self.rank:
                state_dict['state'][key] = {}
            else:
                self.optim.state[param] = recursive_copy_to_device(value, non_blocking=True, device=param.device)
        super().load_state_dict(state_dict)
        OSS._sync_param_groups(state_dict['param_groups'], self.param_groups)
        OSS._sync_param_groups(self.param_groups, self.optim.param_groups)

    def refresh_trainable(self) -> None:
        """Updates the partitioning and communication patterns if the trainability (`requires_grad`)
        of some parameters changed.
        """
        self._default_device = list(self._per_device_params.keys())[0]
        if not hasattr(self, 'optim'):
            self._clear_cache()
            self.optim = self._optim_constructor(self.partition_parameters()[self.rank], **self._optim_defaults)
            OSS._sync_param_groups(self.optim.param_groups, self.param_groups)
        self._setup_flat_buffers()

    def add_param_group(self, param_group: dict) -> None:
        """Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options

        .. warning: This handles updating the shards on all partitions, but needs to be called on all ranks.
        """
        super().add_param_group(param_group)
        if not self.in_super_constructor:
            self._clear_cache()
            param_groups = self.partition_parameters()[self.rank]
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])
            self._setup_flat_buffers()

    @property
    def _local_params(self) -> List[torch.Tensor]:
        """Iterable which goes through the parameters that this rank owns"""
        if self.__local_params is None:
            self.__local_params = list(chain(*[list(filter(lambda x: x.grad is not None, device_params[self.rank])) for device_params in self._per_device_params.values()]))
        return self.__local_params

    @property
    def _param_to_index(self) -> Dict[int, int]:
        """Hash table in between parameter indices in the global optimizer scheme, and the actual params"""
        if len(self.__param_to_index) == 0:
            self.__param_to_index = {id(p): i for i, p in enumerate(chain(*(g['params'] for g in self.param_groups)))}
        return self.__param_to_index

    @property
    def _per_device_params(self) -> Dict[torch.device, List[List[Parameter]]]:
        """Sorted list of all the params, first per device then per rank.

        Within a list params are sorted per number of elements to allow for an easy bucketing.
        """
        if len(self.__per_device_params) == 0:
            for param_group in self.param_groups:
                for param in param_group['params']:
                    device = param.device
                    if self.__per_device_params.get(device) is None:
                        self.__per_device_params[device] = [[] for _ in range(self.world_size)]
                    self.__per_device_params[device][self._param_to_rank[param]] += [param]
            for device in self.__per_device_params.keys():
                for rank_params in self.__per_device_params[device]:
                    rank_params.sort(key=lambda x: x.numel())
        return self.__per_device_params

    @property
    def _param_to_rank(self) -> Dict[torch.Tensor, int]:
        """Map the params to the rank which owns them"""
        if len(self.__param_rank) == 0:
            for rank, param_groups in enumerate(self.partition_parameters()):
                for param_group in param_groups:
                    for param in param_group['params']:
                        self.__param_rank[param] = rank
            logging.debug('FairScale OSS: Parameters dispatched to ranks %s ' % list(self.__param_rank.values()))
        return self.__param_rank

    def _clear_cache(self) -> None:
        self._partition_parameters.clear()
        self.__per_device_params.clear()
        self.__param_rank.clear()
        self.__param_to_index.clear()
        self.__local_params = None

    @staticmethod
    def _sync_param_groups(source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]) -> None:
        """Sync learning rate and other optimizer attributes (needed to support schedulers)."""
        for source_group, destination_group in zip(source, destination):
            for k in filter(lambda x: x != 'params', source_group.keys()):
                destination_group[k] = source_group[k]

    @torch.no_grad()
    def _broadcast_params(self) -> None:
        """Helper function to broadcast all the parameters from a given device"""
        with profiler.record_function('fairscale::oss::refresh_trainable'):
            if torch.device('cuda').type == self._default_device.type:
                for device in self._per_device_params.keys():
                    torch.cuda.synchronize(device=device)
            work_handles = []
            if self.broadcast_fp16:
                for device in self.buckets.keys():
                    for dst_rank, bucket in self.buckets[device].items():
                        bucket.to(dtype=torch.float16, device=device, non_blocking=True, keep_param_alignment=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            for device in self.buckets.keys():
                for dst_rank, bucket in self.buckets[device].items():
                    work_handles.append(dist.broadcast(tensor=bucket.buffer, src=self._local_to_global_rank[dst_rank], group=self.group, async_op=True))
            _ = list(filter(lambda x: x.wait(), work_handles))
            if self.broadcast_fp16:
                for device in self.buckets.keys():
                    for dst_rank, bucket in self.buckets[device].items():
                        bucket.to(dtype=torch.float32, device=device, non_blocking=True, keep_param_alignment=True)

    def _setup_flat_buffers(self) -> None:
        """Make all params which are on the same device and tied to the same rank views of a single buffer.
        This is used at construction time, and anytime parameter trainability is changed (frozen or unfrozen) and
        `refresh_trainability` is called.
        """
        for device, per_rank_params in self._per_device_params.items():
            if device not in self.buckets.keys():
                self.buckets[device] = {}
            for dst_rank, params in enumerate(per_rank_params):
                if len(params) > 0:
                    for param in filter(lambda x: not x.requires_grad, params):
                        param.data = param.data.detach().clone()
                    trainable_params = list(filter(lambda x: x.requires_grad, params))
                    if trainable_params:
                        buffer_size = sum(map(lambda x: x.numel(), trainable_params))
                        bucket = ParamBucket(size=buffer_size, dtype=trainable_params[0].dtype, device=device)
                        for param in trainable_params:
                            bucket.add_param(param)
                        self.buckets[device][dst_rank] = bucket
        devices_in_use = list(self._per_device_params.keys())
        devices_to_pop = list(filter(lambda x: x not in devices_in_use, self.buckets.keys()))
        for d in devices_to_pop:
            self.buckets.pop(d)