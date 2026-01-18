from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _GetCurrentCommand(trace=None, include_separators=True):
    """Returns current command for the purpose of generating help text."""
    if trace:
        current_command = trace.GetCommand(include_separators=include_separators)
    else:
        current_command = ''
    return current_command