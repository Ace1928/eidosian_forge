import os
import pickle
import warnings
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, OrderedDict, Sequence, Set, Tuple, Union
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch._C import _TensorMeta
from torch.nn import Parameter
from typing_extensions import override
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.types import _PATH, _Stateful
class _NotYetLoadedTensor:

    def __init__(self, metatensor: Tensor, archiveinfo: '_LazyLoadingUnpickler', storageinfo: tuple, rebuild_args: tuple) -> None:
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild_from_type_v2(cls, func: Callable, new_type: _TensorMeta, args: tuple, state: dict, *, archiveinfo: Optional['_LazyLoadingUnpickler']=None) -> Any:
        ret = func(*args)
        if isinstance(ret, _NotYetLoadedTensor):
            old_lt = ret._load_tensor

            def _load_tensor() -> Any:
                t = old_lt()
                return torch._tensor._rebuild_from_type_v2(lambda: t, new_type, (), state)
            ret._load_tensor = _load_tensor
            return ret
        return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)

    @classmethod
    def rebuild_parameter(cls, data: Any, requires_grad: bool, backward_hooks: OrderedDict, *, archiveinfo: Optional['_LazyLoadingUnpickler']=None) -> Union[Tensor, '_NotYetLoadedTensor']:
        if isinstance(data, _NotYetLoadedTensor):
            old_lt = data._load_tensor

            def _load_tensor() -> Parameter:
                t = old_lt()
                return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)
            data._load_tensor = _load_tensor
            return data
        return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)

    @classmethod
    def rebuild_tensor_v2(cls, storage: 'TypedStorage', storage_offset: int, size: tuple, stride: tuple, requires_grad: bool, backward_hooks: OrderedDict, metadata: Optional[Any]=None, *, archiveinfo: '_LazyLoadingUnpickler') -> '_NotYetLoadedTensor':
        rebuild_args = (storage_offset, size, stride, requires_grad, backward_hooks, metadata)
        metatensor = torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
        storageinfo = storage.archiveinfo
        return _NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self) -> Tensor:
        from torch.storage import TypedStorage, UntypedStorage
        _, _, fn, _, size = self.storageinfo
        dtype = self.metatensor.dtype
        storage = self.archiveinfo.file_reader.get_storage_from_record(f'data/{fn}', size * torch._utils._element_size(dtype), UntypedStorage)
        uts = storage._typed_storage()._untyped_storage
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            storage = TypedStorage(wrap_storage=uts, dtype=dtype, _internal=True)
        return torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)

    @classmethod
    def __torch_function__(cls, func: Callable, types: Sequence, args: Sequence[Any]=(), kwargs: Optional[Dict]=None) -> Any:
        kwargs = kwargs or {}
        loaded_args = [arg._load_tensor() if isinstance(arg, _NotYetLoadedTensor) else arg for arg in args]
        return func(*loaded_args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in {'dtype', 'grad', 'grad_fn', 'is_meta', 'layout', 'names', 'ndim', 'output_nr', 'requires_grad', 'retains_grad', 'size', 'shape', 'volatile'}:
            return getattr(self.metatensor, name)
        if name in {'contiguous', 'cuda', 'half'}:
            return getattr(self._load_tensor(), name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.metatensor)})'