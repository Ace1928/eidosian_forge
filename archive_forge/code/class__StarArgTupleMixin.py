from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class _StarArgTupleMixin:

    @classmethod
    def _make_homogeneous_tuple(cls, dtype, count):
        return StarArgUniTuple(dtype, count)

    @classmethod
    def _make_heterogeneous_tuple(cls, tys):
        return StarArgTuple(tys)