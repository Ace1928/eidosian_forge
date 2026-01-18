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
def _OneLineResult(result):
    """Returns result serialized to a single line string."""
    if isinstance(result, six.string_types):
        return str(result).replace('\n', ' ')
    if inspect.isfunction(result):
        return '<function {name}>'.format(name=result.__name__)
    if inspect.ismodule(result):
        return '<module {name}>'.format(name=result.__name__)
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result).replace('\n', ' ')