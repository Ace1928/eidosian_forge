from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class DictItemsIterableType(SimpleIterableType):
    """Dictionary iterable type for .items()
    """

    def __init__(self, parent):
        assert isinstance(parent, DictType)
        self.parent = parent
        self.yield_type = self.parent.keyvalue_type
        name = 'items[{}]'.format(self.parent.name)
        self.name = name
        iterator_type = DictIteratorType(self)
        super(DictItemsIterableType, self).__init__(name, iterator_type)