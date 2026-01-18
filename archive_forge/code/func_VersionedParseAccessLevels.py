from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def VersionedParseAccessLevels(path):
    """Parse a YAML representation of a list of Access Levels with basic/custom level conditions.

    Args:
      path: str, path to file containing basic/custom access levels

    Returns:
      list of Access Level objects.

    Raises:
      ParseError: if the file could not be read into the proper object
    """
    data = yaml.load_path(path)
    if not data:
        raise ParseError(path, 'File is empty')
    messages = util.GetMessages(version=api_version)
    message_class = messages.AccessLevel
    try:
        levels = [encoding.DictToMessage(c, message_class) for c in data]
    except Exception as err:
        raise InvalidFormatError(path, six.text_type(err), message_class)
    _ValidateAllLevelFieldsRecognized(path, levels)
    return levels