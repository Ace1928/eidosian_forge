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
def _DictAsString(result, verbose=False):
    """Returns a dict as a string.

  Args:
    result: The dict to convert to a string
    verbose: Whether to include 'hidden' members, those keys starting with _.
  Returns:
    A string representing the dict
  """
    class_attrs = inspectutils.GetClassAttrsDict(result)
    result_visible = {key: value for key, value in result.items() if completion.MemberVisible(result, key, value, class_attrs=class_attrs, verbose=verbose)}
    if not result_visible:
        return '{}'
    longest_key = max((len(str(key)) for key in result_visible.keys()))
    format_string = '{{key:{padding}s}} {{value}}'.format(padding=longest_key + 1)
    lines = []
    for key, value in result.items():
        if completion.MemberVisible(result, key, value, class_attrs=class_attrs, verbose=verbose):
            line = format_string.format(key=str(key) + ':', value=_OneLineResult(value))
            lines.append(line)
    return '\n'.join(lines)