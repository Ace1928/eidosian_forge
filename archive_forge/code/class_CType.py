from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
class CType(ABC):

    @abstractmethod
    def cpp_type(self, *, strip_ref: bool=False) -> str:
        raise NotImplementedError

    @abstractmethod
    def cpp_type_registration_declarations(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def remove_const_ref(self) -> 'CType':
        return self