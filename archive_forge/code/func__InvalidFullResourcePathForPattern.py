from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
from googlecloudsdk.command_lib.scc.hooks import CleanUpUserInput
def _InvalidFullResourcePathForPattern(pattern):
    """Returns an error indicating that provided resource path is invalid."""
    return InvalidSCCInputError("When providing a full resource path, it must match the pattern '%s'." % pattern.pattern)