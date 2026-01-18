import collections
from proto.utils import cached_property
from cloudsdk.google.protobuf.message import Message
@cached_property
def _pb_type(self):
    """Return the protocol buffer type for this sequence."""
    return type(self.pb.GetEntryClass()().value)