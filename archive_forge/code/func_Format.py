import difflib
import math
from ..compat import collections_abc
import six
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format
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