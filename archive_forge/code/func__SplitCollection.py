from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
@classmethod
def _SplitCollection(cls, gri, validate=True):
    """Splits a GRI into its path and collection segments.

    Args:
      gri: str, The GRI string to parse.
      validate: bool, Validate syntax. Use validate=False to handle GRIs under
        construction.

    Returns:
      (str, str), The path and collection parts of the string. The
      collection may be None if not specified in the GRI.

    Raises:
      InvalidGRIFormatException: If the GRI cannot be parsed.
      InvalidGRIPathSyntaxException: If the GRI path cannot be parsed.
    """
    if not gri:
        return (None, None)
    parts = re.split('(?=(?<={)::+[^:}]|(?<=[^:{])::+}|(?<=[^:{])::+[^:}])::', gri)
    if len(parts) > 2:
        raise InvalidGRIFormatException(gri)
    elif len(parts) == 2:
        path, parsed_collection = (parts[0], parts[1])
        if validate:
            cls._ValidateCollection(gri, parsed_collection)
    else:
        path, parsed_collection = (parts[0], None)
    if validate and (path.startswith(':') or path.endswith(':')):
        raise InvalidGRIPathSyntaxException(gri, 'GRIs cannot have empty path segments.')
    return (path, parsed_collection)