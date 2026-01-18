from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from gslib.exception import CommandException
def CreateHelpText(synopsis, description):
    """Helper for adding help text headers given synopsis and description."""
    return SYNOPSIS_PREFIX + synopsis + DESCRIPTION_PREFIX + description