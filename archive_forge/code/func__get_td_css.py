from __future__ import absolute_import, print_function, division
import io
from petl.compat import text_type, numeric_types, next, PY2, izip_longest, \
from petl.errors import ArgumentError
from petl.util.base import Table, Record
from petl.io.base import getcodec
from petl.io.sources import write_source_from_arg
def _get_td_css(h, v, td_styles):
    if td_styles:
        if isinstance(td_styles, string_types):
            return td_styles
        elif callable(td_styles):
            return td_styles(v)
        elif isinstance(td_styles, dict):
            if h in td_styles:
                s = td_styles[h]
                if isinstance(s, string_types):
                    return s
                elif callable(s):
                    return s(v)
                else:
                    raise ArgumentError('expected string or callable, got %r' % s)
        else:
            raise ArgumentError('expected string, callable or dict, got %r' % td_styles)
    if isinstance(v, numeric_types) and (not isinstance(v, bool)):
        return 'text-align: right'
    else:
        return ''