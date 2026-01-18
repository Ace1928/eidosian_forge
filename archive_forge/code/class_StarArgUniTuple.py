from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class StarArgUniTuple(_StarArgTupleMixin, UniTuple):
    """To distinguish from UniTuple() used as argument to a `*args`.
    """