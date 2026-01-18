from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def MessageToFieldPaths(msg):
    """Produce field paths from a message object.

  The result is used to create a FieldMask proto message that contains all field
  paths presented in the object.
  https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/field_mask.proto

  Args:
    msg: A user defined message object that extends the messages.Message class.
    https://github.com/google/apitools/blob/master/apitools/base/protorpclite/messages.py

  Returns:
    The list of field paths.
  """
    fields = []
    for field in msg.all_fields():
        v = msg.get_assigned_value(field.name)
        if field.repeated and (not v):
            continue
        if v is not None:
            if field.name == 'privatePoolV1Config':
                name = 'private_pool_v1_config'
            elif field.name == 'hybridPoolConfig':
                name = 'hybrid_pool_config'
            else:
                name = resource_property.ConvertToSnakeCase(field.name)
            if hasattr(v, 'all_fields'):
                for f in MessageToFieldPaths(v):
                    fields.append('{}.{}'.format(name, f))
            else:
                fields.append(name)
    return fields