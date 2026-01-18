import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
@dataclass
class FakeTensorMeta:
    tensor_size: Tuple[Union[int, torch.SymInt], ...]
    tensor_stride: Tuple[Union[int, torch.SymInt], ...]
    tensor_storage_offset: Union[int, torch.SymInt]
    is_nested: bool

    def size(self) -> Tuple[Union[int, torch.SymInt], ...]:
        return self.tensor_size

    def stride(self) -> Tuple[Union[int, torch.SymInt], ...]:
        return self.tensor_stride

    def storage_offset(self) -> Union[int, torch.SymInt]:
        return self.tensor_storage_offset

    def dim(self) -> int:
        return len(self.tensor_size)

    @staticmethod
    def from_fake(fake) -> 'FakeTensorMeta':
        return FakeTensorMeta(fake.size(), fake.stride(), fake.storage_offset(), fake.is_nested)