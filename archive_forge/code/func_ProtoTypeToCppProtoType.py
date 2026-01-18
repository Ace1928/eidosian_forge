import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
@staticmethod
def ProtoTypeToCppProtoType(proto_type):
    """Converts from a Python proto type to a C++ Proto Type.

    The Python ProtocolBuffer classes specify both the 'Python' datatype and the
    'C++' datatype - and they're not the same. This helper method should
    translate from one to another.

    Args:
      proto_type: the Python proto type (descriptor.FieldDescriptor.TYPE_*)
    Returns:
      int: descriptor.FieldDescriptor.CPPTYPE_*, the C++ type.
    Raises:
      TypeTransformationError: when the Python proto type isn't known.
    """
    try:
        return FieldDescriptor._PYTHON_TO_CPP_PROTO_TYPE_MAP[proto_type]
    except KeyError:
        raise TypeTransformationError('Unknown proto_type: %s' % proto_type)