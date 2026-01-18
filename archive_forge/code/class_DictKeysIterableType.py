from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class DictKeysIterableType(SimpleIterableType):
    """Dictionary iterable type for .keys()
    """

    def __init__(self, parent):
        assert isinstance(parent, DictType)
        self.parent = parent
        self.yield_type = self.parent.key_type
        name = 'keys[{}]'.format(self.parent.name)
        self.name = name
        iterator_type = DictIteratorType(self)
        super(DictKeysIterableType, self).__init__(name, iterator_type)