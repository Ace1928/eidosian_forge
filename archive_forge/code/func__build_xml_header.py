from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def _build_xml_header(style, props, root, head, rows, prologue, encoding):
    tab = _build_nesting(root, False, None) if root else ''
    nested = -1 if style in ('attribute', 'name') else -2
    if head:
        th1 = _build_nesting(head, False, nested)
        col = _build_cols(style, props, head, False)
        th2 = _build_nesting(head, True, nested)
        thd = '{0}\n{1}{2}'.format(th1, col, th2)
    else:
        thd = ''
    tbd = _build_nesting(rows, False, nested)
    if prologue and prologue.startswith('<?xml'):
        thb = '{0}{1}{2}\n'.format(tab, thd, tbd)
        return prologue + thb
    enc = encoding.upper() if encoding else 'UTF-8'
    xml = '<?xml version="1.0" encoding="%s"?>' % enc
    pre = prologue + '\n' if prologue and (not root) else ''
    pos = '\n' + prologue if prologue and root else ''
    res = '{0}\n{1}{2}{3}{4}{5}\n'.format(xml, pre, tab, thd, tbd, pos)
    return res