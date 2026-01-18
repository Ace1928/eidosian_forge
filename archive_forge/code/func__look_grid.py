from __future__ import absolute_import, print_function, division
import locale
from itertools import islice
from collections import defaultdict
from petl.compat import numeric_types, text_type
from petl import config
from petl.util.base import Table
from petl.io.sources import MemorySource
from petl.io.html import tohtml
def _look_grid(table, vrepr, index_header, truncate, width):
    it = iter(table)
    try:
        hdr = next(it)
    except StopIteration:
        return ''
    flds = list(map(text_type, hdr))
    if index_header:
        fldsrepr = ['%s|%s' % (i, r) for i, r in enumerate(flds)]
    else:
        fldsrepr = flds
    rows = list(it)
    rowsrepr = [[vrepr(v) for v in row] for row in rows]
    rowlens = [len(hdr)]
    rowlens.extend([len(row) for row in rows])
    maxrowlen = max(rowlens)
    if len(hdr) < maxrowlen:
        fldsrepr.extend([''] * (maxrowlen - len(hdr)))
    for valsrepr in rowsrepr:
        if len(valsrepr) < maxrowlen:
            valsrepr.extend([''] * (maxrowlen - len(valsrepr)))
    if truncate:
        fldsrepr = [x[:truncate] for x in fldsrepr]
        rowsrepr = [[x[:truncate] for x in valsrepr] for valsrepr in rowsrepr]
    colwidths = [0] * maxrowlen
    for i, fr in enumerate(fldsrepr):
        colwidths[i] = len(fr)
    for valsrepr in rowsrepr:
        for i, vr in enumerate(valsrepr):
            if len(vr) > colwidths[i]:
                colwidths[i] = len(vr)
    sep = '+'
    for w in colwidths:
        sep += '-' * (w + 2)
        sep += '+'
    if width:
        sep = sep[:width]
    sep += '\n'
    hedsep = '+'
    for w in colwidths:
        hedsep += '=' * (w + 2)
        hedsep += '+'
    if width:
        hedsep = hedsep[:width]
    hedsep += '\n'
    fldsline = '|'
    for i, w in enumerate(colwidths):
        f = fldsrepr[i]
        fldsline += ' ' + f
        fldsline += ' ' * (w - len(f))
        fldsline += ' |'
    if width:
        fldsline = fldsline[:width]
    fldsline += '\n'
    rowlines = list()
    for vals, valsrepr in zip(rows, rowsrepr):
        rowline = '|'
        for i, w in enumerate(colwidths):
            vr = valsrepr[i]
            if i < len(vals) and isinstance(vals[i], numeric_types) and (not isinstance(vals[i], bool)):
                rowline += ' ' * (w + 1 - len(vr))
                rowline += vr + ' |'
            else:
                rowline += ' ' + vr
                rowline += ' ' * (w - len(vr))
                rowline += ' |'
        if width:
            rowline = rowline[:width]
        rowline += '\n'
        rowlines.append(rowline)
    output = sep + fldsline + hedsep
    for line in rowlines:
        output += line + sep
    return output