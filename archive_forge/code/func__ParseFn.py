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
def _ParseFn(args):
    """Parses the list of `args` into (varargs, kwargs), remaining_args."""
    kwargs, remaining_kwargs, remaining_args = _ParseKeywordArgs(args, fn_spec)
    parsed_args, kwargs, remaining_args, capacity = _ParseArgs(fn_spec.args, fn_spec.defaults, num_required_args, kwargs, remaining_args, metadata)
    if fn_spec.varargs or fn_spec.varkw:
        capacity = True
    extra_kw = set(kwargs) - set(fn_spec.kwonlyargs)
    if fn_spec.varkw is None and extra_kw:
        raise FireError('Unexpected kwargs present:', extra_kw)
    missing_kwonly = set(required_kwonly) - set(kwargs)
    if missing_kwonly:
        raise FireError('Missing required flags:', missing_kwonly)
    if fn_spec.varargs is not None:
        varargs, remaining_args = (remaining_args, [])
    else:
        varargs = []
    for index, value in enumerate(varargs):
        varargs[index] = _ParseValue(value, None, None, metadata)
    varargs = parsed_args + varargs
    remaining_args += remaining_kwargs
    consumed_args = args[:len(args) - len(remaining_args)]
    return ((varargs, kwargs), consumed_args, remaining_args, capacity)