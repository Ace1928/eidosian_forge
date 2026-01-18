from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding as _encoding
from googlecloudsdk.core import exceptions
import six
@classmethod
def FromErrorPaths(cls, message, errors):
    """Returns a DecodeError from a list of locations of errors.

    Args:
      message: The protorpc Message in which a parsing error occurred.
      errors: List[(edges, field_names)], A list of locations of errors
          encountered while decoding the message.
    """
    type_ = type(message).__name__
    base_msg = 'Failed to parse value(s) in protobuf [{type_}]:'.format(type_=type_)
    error_paths = ['  {type_}.{path}'.format(type_=type_, path=cls._FormatProtoPath(edges, field_names)) for edges, field_names in errors]
    return cls('\n'.join([base_msg] + error_paths))