import collections
import functools
import numbers
import sys
from torch.utils.data.datapipes._hook_iterator import hook_iterator, _SnapshotState
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
from typing import _eval_type, _tp_cache, _type_check, _type_repr  # type: ignore[attr-defined]
from typing import ForwardRef
from abc import ABCMeta
from typing import _GenericAlias  # type: ignore[attr-defined, no-redef]
def _hash_(self):
    return hash((self.__name__, self.type))