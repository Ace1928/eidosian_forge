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
def _GetCallableAvailabilityLines(spec):
    """The list of availability lines for a callable for use in a usage string."""
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    optional_flags = ['--' + flag for flag in itertools.chain(args_with_defaults, _KeywordOnlyArguments(spec, required=False))]
    required_flags = ['--' + flag for flag in _KeywordOnlyArguments(spec, required=True)]
    availability_lines = []
    if optional_flags:
        availability_lines.append(_CreateAvailabilityLine(header='optional flags:', items=optional_flags, header_indent=2))
    if required_flags:
        availability_lines.append(_CreateAvailabilityLine(header='required flags:', items=required_flags, header_indent=2))
    if spec.varkw:
        additional_flags = 'additional flags are accepted' if optional_flags or required_flags else 'flags are accepted'
        availability_lines.append(_CreateAvailabilityLine(header=additional_flags, items=[], header_indent=2))
    return availability_lines