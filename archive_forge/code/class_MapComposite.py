import collections
from proto.utils import cached_property
from cloudsdk.google.protobuf.message import Message
class MapComposite(collections.abc.MutableMapping):
    """A view around a mutable sequence in protocol buffers.

    This implements the full Python MutableMapping interface, but all methods
    modify the underlying field container directly.
    """

    @cached_property
    def _pb_type(self):
        """Return the protocol buffer type for this sequence."""
        return type(self.pb.GetEntryClass()().value)

    def __init__(self, sequence, *, marshal):
        """Initialize a wrapper around a protobuf map.

        Args:
            sequence: A protocol buffers map.
            marshal (~.MarshalRegistry): An instantiated marshal, used to
                convert values going to and from this map.
        """
        self._pb = sequence
        self._marshal = marshal

    def __contains__(self, key):
        return key in tuple(self.keys())

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return self._marshal.to_python(self._pb_type, self.pb[key])

    def __setitem__(self, key, value):
        pb_value = self._marshal.to_proto(self._pb_type, value, strict=True)
        self.pb[key].Clear()
        self.pb[key].MergeFrom(pb_value)

    def __delitem__(self, key):
        self.pb.pop(key)

    def __len__(self):
        return len(self.pb)

    def __iter__(self):
        return iter(self.pb)

    @property
    def pb(self):
        return self._pb