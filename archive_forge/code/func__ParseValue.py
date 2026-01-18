from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import json
import os
import pipes
import re
import shlex
import sys
import types
from fire import completion
from fire import decorators
from fire import formatting
from fire import helptext
from fire import inspectutils
from fire import interact
from fire import parser
from fire import trace
from fire import value_types
from fire.console import console_io
import six
def _ParseValue(value, index, arg, metadata):
    """Parses value, a string, into the appropriate type.

  The function used to parse value is determined by the remaining arguments.

  Args:
    value: The string value to be parsed, typically a command line argument.
    index: The index of the value in the function's argspec.
    arg: The name of the argument the value is being parsed for.
    metadata: Metadata about the function, typically from Fire decorators.
  Returns:
    value, parsed into the appropriate type for calling a function.
  """
    parse_fn = parser.DefaultParseValue
    parse_fns = metadata.get(decorators.FIRE_PARSE_FNS)
    if parse_fns:
        default = parse_fns['default']
        positional = parse_fns['positional']
        named = parse_fns['named']
        if index is not None and 0 <= index < len(positional):
            parse_fn = positional[index]
        elif arg in named:
            parse_fn = named[arg]
        elif default is not None:
            parse_fn = default
    return parse_fn(value)