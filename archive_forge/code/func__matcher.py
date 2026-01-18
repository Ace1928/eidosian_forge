import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def _matcher(it, *args, **kwargs):
    for obj in it:
        try:
            meth = getattr(maybe_evaluate(obj), method)
            reply = on_call(meth, *args, **kwargs) if on_call else meth(*args, **kwargs)
        except AttributeError:
            pass
        else:
            if reply is not None:
                return reply