from collections import namedtuple
from collections.abc import Iterable, Sized
from html import escape as htmlescape
from itertools import chain, zip_longest as izip_longest
from functools import reduce, partial
import io
import re
import math
import textwrap
import dataclasses
def _align_column(strings, alignment, minwidth=0, has_invisible=True, enable_widechars=False, is_multiline=False):
    """[string] -> [padded_string]"""
    strings, padfn = _align_column_choose_padfn(strings, alignment, has_invisible)
    width_fn = _align_column_choose_width_fn(has_invisible, enable_widechars, is_multiline)
    s_widths = list(map(width_fn, strings))
    maxwidth = max(max(_flat_list(s_widths)), minwidth)
    if is_multiline:
        if not enable_widechars and (not has_invisible):
            padded_strings = ['\n'.join([padfn(maxwidth, s) for s in ms.splitlines()]) for ms in strings]
        else:
            s_lens = [[len(s) for s in re.split('[\r\n]', ms)] for ms in strings]
            visible_widths = [[maxwidth - (w - l) for w, l in zip(mw, ml)] for mw, ml in zip(s_widths, s_lens)]
            padded_strings = ['\n'.join([padfn(w, s) for s, w in zip(ms.splitlines() or ms, mw)]) for ms, mw in zip(strings, visible_widths)]
    elif not enable_widechars and (not has_invisible):
        padded_strings = [padfn(maxwidth, s) for s in strings]
    else:
        s_lens = list(map(len, strings))
        visible_widths = [maxwidth - (w - l) for w, l in zip(s_widths, s_lens)]
        padded_strings = [padfn(w, s) for s, w in zip(strings, visible_widths)]
    return padded_strings