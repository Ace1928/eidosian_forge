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