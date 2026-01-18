from googlecloudsdk.command_lib.concepts import concept_managers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.core.util import semver
from googlecloudsdk.core.util import times
import six
class TypeWithSizeConstraint(TypeWithIntervalConstraint):
    """Concept type with size interval constraints.

  Validates that a ConceptType size is within the interval defined by min and
  max endpoints. A missing min or max endpoint indicates that there is no min or
  max size, respectively.
  """
    _DEFAULT_DELIM = ','
    _ALT_DELIM = '^'

    @classmethod
    def _GetIntervalValue(cls, value):
        return len(value) if value else 0

    def __init__(self, name, constraint_kind=None, convert_endpoint=None, convert_value=None, display_endpoint=None, **kwargs):
        super(TypeWithSizeConstraint, self).__init__(name, constraint_kind=constraint_kind or 'size', convert_endpoint=convert_endpoint or int, convert_value=convert_value or self._GetIntervalValue, display_endpoint=convert_endpoint or str, **kwargs)

    def _Split(self, string):
        """Splits string on _DEFAULT_DELIM or the alternate delimiter expression.

    By default, splits on commas:
        'a,b,c' -> ['a', 'b', 'c']

    Alternate delimiter syntax:
        '^:^a,b:c' -> ['a,b', 'c']
        '^::^a:b::c' -> ['a:b', 'c']
        '^,^^a^,b,c' -> ['^a^', ',b', 'c']

    See `gcloud topic escaping` for details.

    Args:
      string: The string with optional alternate delimiter expression.

    Raises:
      exceptions.ParseError: on invalid delimiter expression.

    Returns:
      (string, delimiter) string with the delimiter expression stripped, if any.
    """
        if not string:
            return (None, None)
        delim = self._DEFAULT_DELIM
        if string.startswith(self._ALT_DELIM) and self._ALT_DELIM in string[1:]:
            delim, string = string[1:].split(self._ALT_DELIM, 1)
            if not delim:
                raise exceptions.ParseError(self.GetPresentationName(), 'Invalid delimiter. Please see $ gcloud topic escaping for information on escaping list or dictionary flag values.')
        return (string.split(delim), delim)