import difflib
import math
from ..compat import collections_abc
import six
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
def ProtoEq(a, b):
    """Compares two proto2 objects for equality.

  Recurses into nested messages. Uses list (not set) semantics for comparing
  repeated fields, ie duplicates and order matter.

  Args:
    a: A proto2 message or a primitive.
    b: A proto2 message or a primitive.

  Returns:
    `True` if the messages are equal.
  """

    def Format(pb):
        """Returns a dictionary or unchanged pb bases on its type.

    Specifically, this function returns a dictionary that maps tag
    number (for messages) or element index (for repeated fields) to
    value, or just pb unchanged if it's neither.

    Args:
      pb: A proto2 message or a primitive.
    Returns:
      A dict or unchanged pb.
    """
        if isinstance(pb, message.Message):
            return dict(((desc.number, value) for desc, value in pb.ListFields()))
        elif _IsMap(pb):
            return dict(pb.items())
        elif _IsRepeatedContainer(pb):
            return dict(enumerate(list(pb)))
        else:
            return pb
    a, b = (Format(a), Format(b))
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a == b
    for tag in sorted(set(a.keys()) | set(b.keys())):
        if tag not in a or tag not in b:
            return False
        elif not ProtoEq(a[tag], b[tag]):
            return False
    return True