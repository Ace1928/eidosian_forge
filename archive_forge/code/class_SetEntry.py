from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class SetEntry(Type):
    """
    Internal type class for the entries of a Set's hash table.
    """

    def __init__(self, set_type):
        self.set_type = set_type
        name = 'entry(%s)' % set_type
        super(SetEntry, self).__init__(name)

    @property
    def key(self):
        return self.set_type