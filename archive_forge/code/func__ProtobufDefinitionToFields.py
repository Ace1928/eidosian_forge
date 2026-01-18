from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import property_selector
import six
import six.moves.http_client
def _ProtobufDefinitionToFields(message_class):
    """Flattens the fields in a protocol buffer definition.

  For example, given the following definition:

    message Point {
      required int32 x = 1;
      required int32 y = 2;
      optional string label = 3;
    }

    message Polyline {
      repeated Point point = 1;
      optional string label = 2;
    }

  a call to this function with the Polyline class would produce:

    ['label',
     'point[].label',
     'point[].x',
     'point[].y']

  Args:
    message_class: A class that inherits from protorpc.self.messages.Message
        and defines a protocol buffer.

  Yields:
    The flattened fields, in non-decreasing order.
  """
    for field in sorted(message_class.all_fields(), key=lambda field: field.name):
        if isinstance(field, messages.MessageField):
            for remainder in _ProtobufDefinitionToFields(field.type):
                if field.repeated:
                    yield (field.name + '[].' + remainder)
                else:
                    yield (field.name + '.' + remainder)
        elif field.repeated:
            yield (field.name + '[]')
        else:
            yield field.name