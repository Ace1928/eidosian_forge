from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
def _pytest_fn_decorator(target):
    """Port of langhelpers.decorator with pytest-specific tricks."""
    from sqlalchemy.util.langhelpers import format_argspec_plus
    from sqlalchemy.util.compat import inspect_getfullargspec

    def _exec_code_in_env(code, env, fn_name):
        exec(code, env)
        return env[fn_name]

    def decorate(fn, add_positional_parameters=()):
        spec = inspect_getfullargspec(fn)
        if add_positional_parameters:
            spec.args.extend(add_positional_parameters)
        metadata = dict(__target_fn='__target_fn', __orig_fn='__orig_fn', name=fn.__name__)
        metadata.update(format_argspec_plus(spec, grouped=False))
        code = 'def %(name)s%(grouped_args)s:\n    return %(__target_fn)s(%(__orig_fn)s, %(apply_kw)s)\n' % metadata
        decorated = _exec_code_in_env(code, {'__target_fn': target, '__orig_fn': fn}, fn.__name__)
        if not add_positional_parameters:
            decorated.__defaults__ = getattr(fn, '__func__', fn).__defaults__
            decorated.__wrapped__ = fn
            return update_wrapper(decorated, fn)
        else:
            decorated.__module__ = fn.__module__
            decorated.__name__ = fn.__name__
            if hasattr(fn, 'pytestmark'):
                decorated.pytestmark = fn.pytestmark
            return decorated
    return decorate