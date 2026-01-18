import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
class ProtoEdge(collections.namedtuple('ProtoEdge', ['type_', 'field', 'index'])):
    """A description of a one-level transition from a message to a value.

    Protobuf messages can be arbitrarily nested as fields can be defined with
    any "message" type. This nesting property means that there are often many
    levels of proto messages within a single message instance. This class can
    unambiguously describe a single step from a message to some nested value.

    Properties:
      type_: EdgeType, The type of transition represented by this edge.
      field: str, The name of the message-typed field.
      index: Any, Additional data needed to make the transition. The semantics
          of the "index" property change based on the value of "type_":
            SCALAR: ignored.
            REPEATED: a numeric index into "field"'s list.
            MAP: a key into "field"'s mapping.
    """
    __slots__ = ()

    def __str__(self):
        if self.type_ == EdgeType.SCALAR:
            return self.field
        else:
            return '{}[{}]'.format(self.field, self.index)