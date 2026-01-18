import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _make_cached_stream_func(src_func: t.Callable[[], t.Optional[t.TextIO]], wrapper_func: t.Callable[[], t.TextIO]) -> t.Callable[[], t.Optional[t.TextIO]]:
    cache: t.MutableMapping[t.TextIO, t.TextIO] = WeakKeyDictionary()

    def func() -> t.Optional[t.TextIO]:
        stream = src_func()
        if stream is None:
            return None
        try:
            rv = cache.get(stream)
        except Exception:
            rv = None
        if rv is not None:
            return rv
        rv = wrapper_func()
        try:
            cache[stream] = rv
        except Exception:
            pass
        return rv
    return func