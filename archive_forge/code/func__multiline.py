import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _multiline(self, value, infile, cur_index, maxline):
    """Extract the value, where we are in a multiline situation."""
    quot = value[:3]
    newvalue = value[3:]
    single_line = self._triple_quote[quot][0]
    multi_line = self._triple_quote[quot][1]
    mat = single_line.match(value)
    if mat is not None:
        retval = list(mat.groups())
        retval.append(cur_index)
        return retval
    elif newvalue.find(quot) != -1:
        raise SyntaxError()
    while cur_index < maxline:
        cur_index += 1
        newvalue += '\n'
        line = infile[cur_index]
        if line.find(quot) == -1:
            newvalue += line
        else:
            break
    else:
        raise SyntaxError()
    mat = multi_line.match(line)
    if mat is None:
        raise SyntaxError()
    value, comment = mat.groups()
    return (newvalue + value, comment, cur_index)