from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import re
import textwrap
def _get_or_create_arg_by_name(state, name, is_kwarg=False):
    """Gets or creates a new Arg.

  These Arg objects (Namespaces) are turned into the ArgInfo namedtuples
  returned by parse. Each Arg object is used to collect the name, type, and
  description of a single argument to the docstring's function.

  Args:
    state: The state of the parser.
    name: The name of the arg to create.
    is_kwarg: A boolean representing whether the argument is a keyword arg.
  Returns:
    The new Arg.
  """
    for arg in state.args + state.kwargs:
        if arg.name == name:
            return arg
    arg = Namespace()
    arg.name = name
    arg.type.lines = []
    arg.description.lines = []
    if is_kwarg:
        state.kwargs.append(arg)
    else:
        state.args.append(arg)
    return arg