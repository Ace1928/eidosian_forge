import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def _argsfromspec(spec, replace_defaults=True):
    if spec.defaults:
        split = len(spec.defaults)
        defaults = list(range(len(spec.defaults))) if replace_defaults else spec.defaults
        positional = spec.args[:-split]
        optional = list(zip(spec.args[-split:], defaults))
    else:
        positional, optional = (spec.args, [])
    varargs = spec.varargs
    varkw = spec.varkw
    if spec.kwonlydefaults:
        kwonlyargs = set(spec.kwonlyargs) - set(spec.kwonlydefaults.keys())
        if replace_defaults:
            kwonlyargs_optional = [(kw, i) for i, kw in enumerate(spec.kwonlydefaults.keys())]
        else:
            kwonlyargs_optional = list(spec.kwonlydefaults.items())
    else:
        kwonlyargs, kwonlyargs_optional = (spec.kwonlyargs, [])
    return ', '.join(filter(None, [', '.join(positional), ', '.join((f'{k}={v}' for k, v in optional)), f'*{varargs}' if varargs else None, '*' if (kwonlyargs or kwonlyargs_optional) and (not varargs) else None, ', '.join(kwonlyargs) if kwonlyargs else None, ', '.join((f'{k}="{v}"' for k, v in kwonlyargs_optional)), f'**{varkw}' if varkw else None]))