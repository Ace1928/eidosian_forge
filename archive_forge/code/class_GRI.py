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
class GRI(object):
    """Encapsulates a parsed GRI string.

  Attributes:
    path_fields: [str], The individual fields of the path portion of the GRI.
    collection: str, The collection portion of the GRI.
    is_fully_qualified: bool, True if the original GRI included the collection.
      This could be false if the collection is not defined, or if it was passed
      in explicitly during parsing.
  """

    def __init__(self, path_fields, collection=None, is_fully_qualified=False):
        """Use FromString() to construct a GRI."""
        self.path_fields = path_fields
        self.collection = collection
        self.is_fully_qualified = is_fully_qualified and collection is not None

    def __str__(self):
        gri = ':'.join([self._EscapePathSegment(s) for s in self.path_fields]).rstrip(':')
        if self.is_fully_qualified:
            gri += '::' + self.collection
        return gri

    @classmethod
    def FromString(cls, gri, collection=None, validate=True):
        """Parses a GRI from a string.

    Args:
      gri: str, The GRI to parse.
      collection: str, The collection this GRI is for. If provided and the GRI
        contains a collection, they must match. If not provided, the collection
        in the GRI will be used, or None if it is not specified.
      validate: bool, Validate syntax. Use validate=False to handle GRIs under
        construction.

    Returns:
      A parsed GRI object.

    Raises:
      GRICollectionMismatchException: If the given collection does not match the
        collection specified in the GRI.
    """
        path, parsed_collection = cls._SplitCollection(gri, validate=validate)
        if not collection:
            collection = parsed_collection
        elif validate:
            cls._ValidateCollection(gri, collection)
            if parsed_collection and parsed_collection != collection:
                raise GRICollectionMismatchException(gri, expected_collection=collection, parsed_collection=parsed_collection)
        path_fields = cls._SplitPath(path)
        return GRI(path_fields, collection, is_fully_qualified=bool(parsed_collection))

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

    @classmethod
    def _ValidateCollection(cls, gri, collection):
        if not re.match('^\\w+\\.\\w+(?:\\.\\w+)*$', collection):
            raise InvalidGRICollectionSyntaxException(gri, collection)

    @classmethod
    def _SplitPath(cls, path):
        """Splits a GRI into its individual path segments.

    Args:
      path: str, The path segment of the GRI (from _SplitCollection)

    Returns:
      [str], A list of the path segments of the GRI.
    """
        if not path:
            return []
        parts = re.split('(?=(?<={):+[^:}]|(?<=[^:{]):+}|(?<=[^:{]):+[^:}]):', path)
        return [cls._UnescapePathSegment(part) for part in parts]

    @classmethod
    def _UnescapePathSegment(cls, segment):
        return re.sub('{(:+)}', '\\1', segment)

    @classmethod
    def _EscapePathSegment(cls, segment):
        return re.sub('(:+)', '{\\1}', segment)