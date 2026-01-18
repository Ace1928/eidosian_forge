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
def _GetCallableUsageItems(spec, metadata):
    """A list of elements that comprise the usage summary for a callable."""
    args_with_no_defaults = spec.args[:len(spec.args) - len(spec.defaults)]
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    accepts_positional_args = metadata.get(decorators.ACCEPTS_POSITIONAL_ARGS)
    if not accepts_positional_args:
        items = ['--{arg}={upper}'.format(arg=arg, upper=arg.upper()) for arg in args_with_no_defaults]
    else:
        items = [arg.upper() for arg in args_with_no_defaults]
    if args_with_defaults or spec.kwonlyargs or spec.varkw:
        items.append('<flags>')
    if spec.varargs:
        items.append('[{varargs}]...'.format(varargs=spec.varargs.upper()))
    return items