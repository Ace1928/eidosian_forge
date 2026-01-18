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
class _Listener(object):
    """MessageListener implementation that a parent message registers with its
  child message.

  In order to support semantics like:

    foo.bar.baz.moo = 23
    assert foo.HasField('bar')

  ...child objects must have back references to their parents.
  This helper class is at the heart of this support.
  """

    def __init__(self, parent_message):
        """Args:
      parent_message: The message whose _Modified() method we should call when
        we receive Modified() messages.
    """
        if isinstance(parent_message, weakref.ProxyType):
            self._parent_message_weakref = parent_message
        else:
            self._parent_message_weakref = weakref.proxy(parent_message)
        self.dirty = False

    def Modified(self):
        if self.dirty:
            return
        try:
            self._parent_message_weakref._Modified()
        except ReferenceError:
            pass