import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
class PipelineHelper(object):
    """
    PipelineHelper assists with implementing pipelining in model parallelism.

    For a tutorial on model parallelism, as it's implemented in parts of ParlAI,
    see https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html.

    Usage:
    >>> my_model = PipelineHelper().make_parallel(my_model)

    Note that you will need to manually implement logic which handles the
    moved layers.
    """

    def __init__(self):
        self.__device_allocations = {}
        self.num_devices = torch.cuda.device_count()
        self.devices = []
        for i in range(self.num_devices):
            d = f'cuda:{i}'
            self.devices.append(d)
            self.__device_allocations[d] = 0

    def make_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Allocate specific layers in a model to be ModelParallel.

        Limited to only ModuleLists within the model.  Uses some heuristics to
        attempt to evenly distribute layers across GPUs, in order to balance
        memory usage. They are:

        - Assume the 0th GPU will host the optimizer, word embeddings, etc.
        - Assume activation memory is linear with the number of parameters.
        - All layers are approximately equal in size.
        """
        self.__device_allocations['cuda:0'] += trainable_parameters(model) * 3
        model.apply(self._place_modulelist)
        model._apply(self._move_rest_to_cuda0)
        return model

    def _move_rest_to_cuda0(self, parameter: torch.Tensor):
        if parameter.device.type == 'cpu':
            return parameter.to('cuda:0')
        else:
            return parameter

    def _place_modulelist(self, submodule: torch.nn.Module) -> None:
        if not isinstance(submodule, torch.nn.ModuleList):
            return
        if getattr(submodule, 'model_parallel_exempt', False):
            return
        assert isinstance(submodule, torch.nn.ModuleList)
        layers = submodule
        layers.is_model_parallel = True
        keyfunc = self.__device_allocations.__getitem__
        layer_assignments = {k: 0 for k in self.devices}
        for layer_no, layer in enumerate(layers):
            if layer_no == 0:
                mostfree = 'cuda:0'
            else:
                mostfree = min(self.devices, key=keyfunc)
            self.__device_allocations[mostfree] += trainable_parameters(layer) * 32
            layer_assignments[mostfree] += 1
        devices = [d for i, d in enumerate(self.devices[:]) if layer_assignments[d] > 0]
        for layer_no, layer in enumerate(layers):
            layer_gpu = devices[0]
            assert layer_assignments[layer_gpu] > 0
            logging.debug(f'Model Parallel: Assigning {layer_no} to {layer_gpu}')
            layer._mp_gpu = layer_gpu
            layers[layer_no] = layer.to(layer_gpu)
            layer_assignments[layer_gpu] -= 1
            if layer_assignments[layer_gpu] == 0:
                devices.pop(0)

    @staticmethod
    def guess_split_size(item: Chunk, num_gpus: Optional[int]=None, dim=0) -> int:
        """
        Estimate the number of chunks we should split the batch into via heuristics.
        """
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        if isinstance(item, torch.Tensor):
            if num_gpus == 1:
                return item.size(dim)
            return max(1, item.size(dim) // int(num_gpus * 2))
        elif isinstance(item, tuple):
            return PipelineHelper.guess_split_size(item[0], num_gpus)
        elif isinstance(item, dict):
            return PipelineHelper.guess_split_size(list(item.values())[0], num_gpus)
        raise TypeError(f'Cannot determine split size for {type(item)}')

    @staticmethod
    def split(item: Chunk, split_size: Optional[int]=None, dim=0) -> List[Chunk]:
        """
        Split a tensor or group of tensors into smaller chunks of the same type.

        :param item:
            The item being split. May be a Tensor, a tuple of Tensors, or a
            dictionary mapping str -> Tensor.
        :param split_size:
            The maximum size of each output chunk. If None, we will guess using
            heuristics
        :param dim:
            The dimension to split along.
        """
        if split_size is None:
            split_size = PipelineHelper.guess_split_size(item)
        if isinstance(item, torch.Tensor):
            return list(torch.split(item, split_size, dim))
        elif isinstance(item, tuple):
            return list(zip(*(PipelineHelper.split(i, split_size, dim) for i in item)))
        elif isinstance(item, dict):
            if item == {}:
                return itertools.repeat({})
            if {} in [x for x in item.values() if isinstance(x, dict)]:
                raise ValueError('Cannot handle a dictionary with an empty dictionary inside.')
            if () in [x for x in item.values() if isinstance(x, tuple)]:
                raise ValueError('Cannot handle a dictionary with an empty tuple inside.')
            d = {k: PipelineHelper.split(v, split_size, dim) for k, v in item.items()}
            return [dict(zip(d.keys(), values)) for values in zip(*(d[k] for k in d.keys()))]
        else:
            raise TypeError(f'Cannot split type {type(item)}')

    @staticmethod
    def join(items: List[Chunk], dim=0) -> Chunk:
        """
        Join chunks back together, the inverse of split.

        :param items:
            All the output chunks. Each chunk may be a tensor or a group of
            tensors.
        :param dim:
            The dimension to join along.
        """
        if len(items) == 0:
            raise IndexError('Cannot rejoin an empty list of chunks.')
        item0 = items[0]
        if isinstance(item0, torch.Tensor):
            return torch.cat(items, dim=dim)
        elif isinstance(item0, tuple):
            return tuple((PipelineHelper.join(x, dim=dim) for x in zip(*items)))
        elif isinstance(item0, dict):
            keys = item0.keys()
            return {k: PipelineHelper.join([c[k] for c in items], dim=dim) for k in keys}
        else:
            raise TypeError(f'Cannot join list of type {type(item0)}')

    @staticmethod
    def chunk_to(chunk: Chunk, device: str) -> Chunk:
        """
        Move the chunk to the device.

        Handles chunks which are groups of tensors.
        """
        if isinstance(chunk, torch.Tensor):
            return chunk.to(device)
        elif isinstance(chunk, tuple):
            return tuple((PipelineHelper.chunk_to(c, device) for c in chunk))
        elif isinstance(chunk, dict):
            return {k: PipelineHelper.chunk_to(v, device) for k, v in chunk.items()}
        else:
            raise TypeError('chunk_to only compatible with tensors, tuples or dicts.')

    @staticmethod
    def schedule_work_items(layers: torch.nn.ModuleList, chunks: List[Chunk]):
        """
        Iterate through chunks and layers that should be pipelined.

        Each iteration of this generator yields the following properties:

            - layer_nos: a list of indices of layers for you to forward through
            - chunk_idx: the index of the chunk we are manipulating. Use this
              if you need to update chunk representations.
            - next_device: where the chunk should be moved to AFTER the layer
              computation is done.
        """
        num_chunks = len(chunks)
        for l in layers:
            if not hasattr(l, '_mp_gpu'):
                raise RuntimeError('You must run PipelineHelper.make_parallel on the ModuleList before you can use iterate_layers_chunks.')
        devices = {device_idx: (dev, list(grp)) for device_idx, (dev, grp) in enumerate(itertools.groupby(range(len(layers)), lambda x: layers[x]._mp_gpu))}
        num_timesteps = len(devices) + num_chunks
        for timestep in range(num_timesteps):
            for chunk_idx in range(num_chunks):
                device_idx = timestep - chunk_idx
                if device_idx >= 0 and device_idx < len(devices):
                    dev, layers_nos = devices[device_idx]
                    next_device, _ = devices[(device_idx + 1) % len(devices)]
                    assert device_idx in devices
                    yield PipelineWorkItem(chunk_idx=chunk_idx, layer_nos=layers_nos, next_device=next_device)