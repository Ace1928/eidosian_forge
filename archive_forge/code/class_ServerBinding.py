from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServerBinding(_messages.Message):
    """A ServerBinding object.

  Enums:
    TypeValueValuesEnum:

  Fields:
    type: A TypeValueValuesEnum attribute.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """TypeValueValuesEnum enum type.

    Values:
      RESTART_NODE_ON_ANY_SERVER: Node may associate with any physical server
        over its lifetime.
      RESTART_NODE_ON_MINIMAL_SERVERS: Node may associate with minimal
        physical servers over its lifetime.
      SERVER_BINDING_TYPE_UNSPECIFIED: <no description>
    """
        RESTART_NODE_ON_ANY_SERVER = 0
        RESTART_NODE_ON_MINIMAL_SERVERS = 1
        SERVER_BINDING_TYPE_UNSPECIFIED = 2
    type = _messages.EnumField('TypeValueValuesEnum', 1)