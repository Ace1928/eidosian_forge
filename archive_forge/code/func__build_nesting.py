from __future__ import absolute_import, print_function, division
from operator import attrgetter
import itertools
from petl.compat import string_types, text_type
from petl.util.base import Table, fieldnames, iterpeek
from petl.io.sources import read_source_from_arg
from petl.io.text import totext
def _build_nesting(path, closing, index):
    if not path:
        return ''
    fmt = '</%s>' if closing else '<%s>'
    if '/' not in path:
        return fmt % path
    parts = path.split('/')
    elements = parts[0:index] if index else parts
    if closing:
        elements.reverse()
    tags = [fmt % e for e in elements]
    return ''.join(tags)