import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class BackendMetadata:
    kernel: str
    structured: bool
    cpp_namespace: str

    def supports_symint(self) -> bool:
        return '_symint' in self.kernel