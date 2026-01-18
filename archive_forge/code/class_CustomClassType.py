import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class CustomClassType(Type):
    class_name: str

    def __str__(self) -> str:
        """
        Return the class name will prefix __torch__.torch.classes
        """
        return f'__torch__.torch.classes.{self.class_name}'

    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        return False

    def is_symint_like(self) -> bool:
        return False

    def is_nullable(self) -> bool:
        """
        Assume a custom class is not nullable.
        """
        return False

    def is_list_like(self) -> Optional['ListType']:
        return None