from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def _fmt_arglist(self, args, npoargs=0, npargs=0, pargs=None, nkargs=0, kargs=None, hide_self=False):
    arglist = []
    for arg in args:
        if not hide_self or not arg.entry.is_self_arg:
            arg_doc = self._fmt_arg(arg)
            arglist.append(arg_doc)
    if pargs:
        arg_doc = self._fmt_star_arg(pargs)
        arglist.insert(npargs + npoargs, '*%s' % arg_doc)
    elif nkargs:
        arglist.insert(npargs + npoargs, '*')
    if npoargs:
        arglist.insert(npoargs, '/')
    if kargs:
        arg_doc = self._fmt_star_arg(kargs)
        arglist.append('**%s' % arg_doc)
    return arglist