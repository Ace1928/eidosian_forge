from __future__ import annotations
import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path
from .caching import (  # noqa: F401
from .compression import compr
from .registry import filesystem, get_filesystem_class
from .utils import (
def _un_chain(path, kwargs):
    x = re.compile('.*[^a-z]+.*')
    bits = [p if '://' in p or x.match(p) else p + '://' for p in path.split('::')] if '::' in path else [path]
    out = []
    previous_bit = None
    kwargs = kwargs.copy()
    for bit in reversed(bits):
        protocol = kwargs.pop('protocol', None) or split_protocol(bit)[0] or 'file'
        cls = get_filesystem_class(protocol)
        extra_kwargs = cls._get_kwargs_from_urls(bit)
        kws = kwargs.pop(protocol, {})
        if bit is bits[0]:
            kws.update(kwargs)
        kw = dict(**extra_kwargs, **kws)
        bit = cls._strip_protocol(bit)
        if protocol in {'blockcache', 'filecache', 'simplecache'} and 'target_protocol' not in kw:
            bit = previous_bit
        out.append((bit, protocol, kw))
        previous_bit = bit
    out.reverse()
    return out