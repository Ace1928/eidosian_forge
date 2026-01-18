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
def _GetArgDefault(flag, spec):
    """Returns a string describing a flag's default value.

  Args:
    flag: The name of the flag.
    spec: An instance of fire.inspectutils.FullArgSpec, containing type and
     default information about the arguments to a callable.
  Returns:
    A string to be used in constructing the help screen for the function, the
    empty string if the flag does not have a default or the default is not
    available.
  """
    num_defaults = len(spec.defaults)
    args_with_defaults = spec.args[-num_defaults:]
    for arg, default in zip(args_with_defaults, spec.defaults):
        if arg == flag:
            return repr(default)
    if flag in spec.kwonlydefaults:
        return repr(spec.kwonlydefaults[flag])
    return ''