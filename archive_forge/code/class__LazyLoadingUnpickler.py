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
class _LazyLoadingUnpickler(pickle.Unpickler):

    def __init__(self, file: IO, file_reader: torch.PyTorchFileReader) -> None:
        super().__init__(file)
        self.file_reader = file_reader

    @override
    def find_class(self, module: str, name: str) -> Any:
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return partial(_NotYetLoadedTensor.rebuild_tensor_v2, archiveinfo=self)
        if module == 'torch._tensor' and name == '_rebuild_from_type_v2':
            return partial(_NotYetLoadedTensor.rebuild_from_type_v2, archiveinfo=self)
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return partial(_NotYetLoadedTensor.rebuild_parameter, archiveinfo=self)
        return super().find_class(module, name)

    @override
    def persistent_load(self, pid: tuple) -> 'TypedStorage':
        from torch.storage import TypedStorage
        _, cls, _, _, _ = pid
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            storage = TypedStorage(dtype=cls().dtype, device='meta')
        storage.archiveinfo = pid
        return storage