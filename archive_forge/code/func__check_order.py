import itertools
import warnings
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle
def _check_order(self, handle: FlatParamHandle, is_training: bool) -> None:
    """
        Checks the forward execution order as long as ``is_training`` is
        ``True`` since checking in eval mode is not supported. This only checks
        if the distributed debug level is DETAIL.

        - On the first iteration, this uses all-gathers to check that all ranks
        are all-gathering the same handles and hence ``FlatParameter`` s,
        raising an error if not.
        - On subsequent iterations, this checks that each rank is locally
        consistent with its own forward order from the first iteration, issuing
        a warning if not. This issues a warning on the first deviating
        iteration and stops warning thereafter.
        """
    if not is_training or not self._checking_order:
        return
    if self.is_first_iter:
        msg_prefix = 'Forward order differs across ranks:'
        optional_local_indices: Tuple[Optional[int], ...] = self._get_handle_indices(handle)
        device = handle.device
        num_valid_indices = sum((index is not None for index in optional_local_indices))
        tensor_kwargs: Dict[str, Union[torch.dtype, torch.device]] = {'dtype': torch.int32, 'device': device}
        world_num_valid_indices = torch.zeros(self.world_size, **tensor_kwargs)
        local_num_valid_indices = torch.tensor([num_valid_indices], **tensor_kwargs)
        dist.all_gather_into_tensor(world_num_valid_indices, local_num_valid_indices, group=self.process_group)
        world_num_valid_indices = world_num_valid_indices.cpu()
        assert self.world_size is not None
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            for (r1, n1), (r2, n2) in itertools.combinations(((rank, world_num_valid_indices[rank]) for rank in range(self.world_size)), 2):
                if n1 != n2:
                    raise RuntimeError(f'{msg_prefix} rank {r1} is all-gathering {n1} parameters while rank {r2} is all-gathering {n2} parameters')
        world_indices = torch.zeros(self.world_size * num_valid_indices, **tensor_kwargs)
        local_indices = torch.tensor(optional_local_indices, **tensor_kwargs)
        dist.all_gather_into_tensor(world_indices, local_indices, group=self.process_group)
        world_indices = world_indices.cpu()
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            for (r1, i1), (r2, i2) in itertools.combinations(((rank, world_indices[rank * num_valid_indices:(rank + 1) * num_valid_indices]) for rank in range(self.world_size)), 2):
                if i1 != i2:
                    r1_param_names = self._get_names_from_handle_indices(i1)
                    r2_param_names = self._get_names_from_handle_indices(i2)
                    raise RuntimeError(f'{msg_prefix} rank {r1} is all-gathering parameters for {r1_param_names} while rank {r2} is all-gathering parameters for {r2_param_names}')
    else:
        if self.warn_status == _ExecOrderWarnStatus.WARNED:
            return
        msg_prefix = None
        if self.current_order_index >= len(self.handles_pre_forward_order):
            msg_prefix = 'Expected to not all-gather any more parameters in the forward but trying to all-gather parameters for '
        else:
            expected_handle = self.handles_pre_forward_order[self.current_order_index]
            if expected_handle != handle:
                expected_param_names = self._get_names_from_handles(expected_handle)
                msg_prefix = f'Expected to all-gather for {expected_param_names} but trying to all-gather parameters for '
        if msg_prefix is not None:
            param_names = self._get_names_from_handles(handle)
            msg_suffix = f'{param_names}' if param_names else 'a newly-added parameter since construction time'
            warnings.warn(f'Forward order differs from that of the first iteration on rank {self.rank}. Collectives are unchecked and may give incorrect results or hang.\n{msg_prefix}{msg_suffix}')
            self.warn_status = _ExecOrderWarnStatus.WARNING
        self.current_order_index += 1