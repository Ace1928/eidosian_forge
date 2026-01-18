from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding as _encoding
from googlecloudsdk.core import exceptions
import six
def DictToMessageWithErrorCheck(dict_, message_type, throw_on_unexpected_fields=True):
    """Convert "dict_" to a message of type message_type and check for errors.

  A common use case is to define the dictionary by deserializing yaml or json.

  Args:
    dict_: The dict to parse into a protorpc Message.
    message_type: The protorpc Message type.
    throw_on_unexpected_fields: If this flag is set, an error will be raised if
    the dictionary contains unrecognized fields.

  Returns:
    A message of type "message_type" parsed from "dict_".

  Raises:
    DecodeError: One or more unparsable values were found in the parsed message.
  """
    try:
        message = _encoding.DictToMessage(dict_, message_type)
    except _messages.ValidationError as e:
        raise ScalarTypeMismatchError('Failed to parse value in protobuf [{type_}]:\n  {type_}.??: "{msg}"'.format(type_=message_type.__name__, msg=six.text_type(e)))
    except AttributeError:
        raise
    else:
        errors = list(_encoding.UnrecognizedFieldIter(message))
        if errors and throw_on_unexpected_fields:
            raise DecodeError.FromErrorPaths(message, errors)
        return message