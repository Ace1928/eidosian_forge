from collections.abc import Iterable
from collections.abc import Sequence as pySequence
from types import MappingProxyType
from .abstract import (
from .common import (
from .misc import Undefined, unliteral, Optional, NoneType
from ..typeconv import Conversion
from ..errors import TypingError
from .. import utils
class BaseContainerIterator(SimpleIteratorType):
    """
    Convenience base class for some container iterators.

    Derived classes must implement the *container_class* attribute.
    """

    def __init__(self, container):
        assert isinstance(container, self.container_class), container
        self.container = container
        yield_type = container.dtype
        name = 'iter(%s)' % container
        super(BaseContainerIterator, self).__init__(name, yield_type)

    def unify(self, typingctx, other):
        cls = type(self)
        if isinstance(other, cls):
            container = typingctx.unify_pairs(self.container, other.container)
            if container is not None:
                return cls(container)

    @property
    def key(self):
        return self.container