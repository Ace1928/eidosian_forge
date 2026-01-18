import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_function_decl(self, tp, name):
    assert isinstance(tp, model.FunctionPtrType)
    if tp.ellipsis:
        self._generate_gen_const(False, name, tp)
        return
    prnt = self._prnt
    numargs = len(tp.args)
    argnames = []
    for i, type in enumerate(tp.args):
        indirection = ''
        if isinstance(type, model.StructOrUnion):
            indirection = '*'
        argnames.append('%sx%d' % (indirection, i))
    context = 'argument of %s' % name
    arglist = [type.get_c_name(' %s' % arg, context) for type, arg in zip(tp.args, argnames)]
    tpresult = tp.result
    if isinstance(tpresult, model.StructOrUnion):
        arglist.insert(0, tpresult.get_c_name(' *r', context))
        tpresult = model.void_type
    arglist = ', '.join(arglist) or 'void'
    wrappername = '_cffi_f_%s' % name
    self.export_symbols.append(wrappername)
    if tp.abi:
        abi = tp.abi + ' '
    else:
        abi = ''
    funcdecl = ' %s%s(%s)' % (abi, wrappername, arglist)
    context = 'result of %s' % name
    prnt(tpresult.get_c_name(funcdecl, context))
    prnt('{')
    if isinstance(tp.result, model.StructOrUnion):
        result_code = '*r = '
    elif not isinstance(tp.result, model.VoidType):
        result_code = 'return '
    else:
        result_code = ''
    prnt('  %s%s(%s);' % (result_code, name, ', '.join(argnames)))
    prnt('}')
    prnt()