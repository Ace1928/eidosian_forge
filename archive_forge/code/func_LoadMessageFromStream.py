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
def LoadMessageFromStream(stream, msg_type, msg_friendly_name, skip_camel_case=None, path=None):
    """Load a proto message from a stream of JSON or YAML text.

  Args:
    stream: file-like object containing the JSON or YAML data to be decoded.
    msg_type: The protobuf message type to create.
    msg_friendly_name: A readable name for the message type, for use in error
      messages.
    skip_camel_case: Contains proto field names or map keys whose values should
      not have camel case applied.
    path: str or None. Optional path to be used in error messages.

  Raises:
    ParserError: If there was a problem parsing the stream as a dict.
    ParseProtoException: If there was a problem interpreting the stream as the
    given message type.

  Returns:
    Proto message, The message that got decoded.
  """
    if skip_camel_case is None:
        skip_camel_case = []
    try:
        structured_data = yaml.load(stream, file_hint=path)
    except yaml.Error as e:
        raise cloudbuild_exceptions.ParserError(path, e.inner_error)
    if not isinstance(structured_data, dict):
        raise cloudbuild_exceptions.ParserError(path, 'Could not parse as a dictionary.')
    return _YamlToMessage(structured_data, msg_type, msg_friendly_name, skip_camel_case, path)