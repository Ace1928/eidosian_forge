from __future__ import annotations
import builtins
import collections.abc as collections_abc
import re
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import ForwardRef
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import compat
def _copy_generic_annotation_with(annotation: GenericProtocol[_T], elements: Tuple[_AnnotationScanType, ...]) -> Type[_T]:
    if hasattr(annotation, 'copy_with'):
        return annotation.copy_with(elements)
    else:
        return annotation.__origin__[elements]