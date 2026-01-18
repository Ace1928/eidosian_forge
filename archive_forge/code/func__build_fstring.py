from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def _build_fstring(self, pos, ustring, format_args):
    args = iter(format_args)
    substrings = []
    can_be_optimised = True
    for s in re.split(self._parse_string_format_regex, ustring):
        if not s:
            continue
        if s == u'%%':
            substrings.append(ExprNodes.UnicodeNode(pos, value=EncodedString(u'%'), constant_result=u'%'))
            continue
        if s[0] != u'%':
            if s[-1] == u'%':
                warning(pos, "Incomplete format: '...%s'" % s[-3:], level=1)
                can_be_optimised = False
            substrings.append(ExprNodes.UnicodeNode(pos, value=EncodedString(s), constant_result=s))
            continue
        format_type = s[-1]
        try:
            arg = next(args)
        except StopIteration:
            warning(pos, 'Too few arguments for format placeholders', level=1)
            can_be_optimised = False
            break
        if arg.is_starred:
            can_be_optimised = False
            break
        if format_type in u'asrfdoxX':
            format_spec = s[1:]
            conversion_char = None
            if format_type in u'doxX' and u'.' in format_spec:
                can_be_optimised = False
            elif format_type in u'ars':
                format_spec = format_spec[:-1]
                conversion_char = format_type
                if format_spec.startswith('0'):
                    format_spec = '>' + format_spec[1:]
            elif format_type == u'd':
                conversion_char = 'd'
            if format_spec.startswith('-'):
                format_spec = '<' + format_spec[1:]
            substrings.append(ExprNodes.FormattedValueNode(arg.pos, value=arg, conversion_char=conversion_char, format_spec=ExprNodes.UnicodeNode(pos, value=EncodedString(format_spec), constant_result=format_spec) if format_spec else None))
        else:
            can_be_optimised = False
            break
    if not can_be_optimised:
        return None
    try:
        next(args)
    except StopIteration:
        pass
    else:
        warning(pos, 'Too many arguments for format placeholders', level=1)
        return None
    node = ExprNodes.JoinedStrNode(pos, values=substrings)
    return self.visit_JoinedStrNode(node)