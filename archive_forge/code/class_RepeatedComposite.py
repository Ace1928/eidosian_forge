import collections
import copy
from proto.utils import cached_property
class RepeatedComposite(Repeated):
    """A view around a mutable sequence of messages in protocol buffers.

    This implements the full Python MutableSequence interface, but all methods
    modify the underlying field container directly.
    """

    @cached_property
    def _pb_type(self):
        """Return the protocol buffer type for this sequence."""
        if self._proto_type is not None:
            return self._proto_type
        if len(self.pb) > 0:
            return type(self.pb[0])
        if hasattr(self.pb, '_message_descriptor') and hasattr(self.pb._message_descriptor, '_concrete_class'):
            return self.pb._message_descriptor._concrete_class
        canary = copy.deepcopy(self.pb).add()
        return type(canary)

    def __eq__(self, other):
        if super().__eq__(other):
            return True
        return tuple([i for i in self]) == tuple(other)

    def __getitem__(self, key):
        return self._marshal.to_python(self._pb_type, self.pb[key])

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if -len(self) <= key < len(self):
                self.pop(key)
                self.insert(key, value)
            else:
                raise IndexError('list assignment index out of range')
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if not isinstance(value, collections.abc.Iterable):
                raise TypeError('can only assign an iterable')
            if step == 1:
                for index, item in enumerate(value):
                    if start + index < stop:
                        self.pop(start + index)
                    self.insert(start + index, item)
                for _ in range(stop - start - len(value)):
                    self.pop(start + len(value))
            else:
                indices = range(start, stop, step)
                if len(value) != len(indices):
                    raise ValueError(f'attempt to assign sequence of size {len(value)} to extended slice of size {len(indices)}')
                for index, item in zip(indices, value):
                    self[index] = item
        else:
            raise TypeError(f'list indices must be integers or slices, not {type(key).__name__}')

    def insert(self, index: int, value):
        """Insert ``value`` in the sequence before ``index``."""
        pb_value = self._marshal.to_proto(self._pb_type, value)
        self.pb.insert(index, pb_value)