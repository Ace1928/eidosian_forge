from io import BytesIO
import struct
import sys
import warnings
import weakref
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
def _ReraiseTypeErrorWithFieldName(message_name, field_name):
    """Re-raise the currently-handled TypeError with the field name added."""
    exc = sys.exc_info()[1]
    if len(exc.args) == 1 and type(exc) is TypeError:
        exc = TypeError('%s for field %s.%s' % (str(exc), message_name, field_name))
    raise exc.with_traceback(sys.exc_info()[2])