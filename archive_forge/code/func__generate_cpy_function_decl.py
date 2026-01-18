import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _generate_cpy_function_decl(self, tp, name):
    assert not self.target_is_python
    assert isinstance(tp, model.FunctionPtrType)
    if tp.ellipsis:
        self._generate_cpy_constant_decl(tp, name)
        return
    prnt = self._prnt
    numargs = len(tp.args)
    if numargs == 0:
        argname = 'noarg'
    elif numargs == 1:
        argname = 'arg0'
    else:
        argname = 'args'
    arguments = []
    call_arguments = []
    context = 'argument of %s' % name
    for i, type in enumerate(tp.args):
        arguments.append(type.get_c_name(' x%d' % i, context))
        call_arguments.append('x%d' % i)
    repr_arguments = ', '.join(arguments)
    repr_arguments = repr_arguments or 'void'
    if tp.abi:
        abi = tp.abi + ' '
    else:
        abi = ''
    name_and_arguments = '%s_cffi_d_%s(%s)' % (abi, name, repr_arguments)
    prnt('static %s' % (tp.result.get_c_name(name_and_arguments),))
    prnt('{')
    call_arguments = ', '.join(call_arguments)
    result_code = 'return '
    if isinstance(tp.result, model.VoidType):
        result_code = ''
    prnt('  %s%s(%s);' % (result_code, name, call_arguments))
    prnt('}')
    prnt('#ifndef PYPY_VERSION')
    prnt('static PyObject *')
    prnt('_cffi_f_%s(PyObject *self, PyObject *%s)' % (name, argname))
    prnt('{')
    context = 'argument of %s' % name
    for i, type in enumerate(tp.args):
        arg = type.get_c_name(' x%d' % i, context)
        prnt('  %s;' % arg)
    localvars = set()
    freelines = set()
    for type in tp.args:
        self._extra_local_variables(type, localvars, freelines)
    for decl in sorted(localvars):
        prnt('  %s;' % (decl,))
    if not isinstance(tp.result, model.VoidType):
        result_code = 'result = '
        context = 'result of %s' % name
        result_decl = '  %s;' % tp.result.get_c_name(' result', context)
        prnt(result_decl)
        prnt('  PyObject *pyresult;')
    else:
        result_decl = None
        result_code = ''
    if len(tp.args) > 1:
        rng = range(len(tp.args))
        for i in rng:
            prnt('  PyObject *arg%d;' % i)
        prnt()
        prnt('  if (!PyArg_UnpackTuple(args, "%s", %d, %d, %s))' % (name, len(rng), len(rng), ', '.join(['&arg%d' % i for i in rng])))
        prnt('    return NULL;')
    prnt()
    for i, type in enumerate(tp.args):
        self._convert_funcarg_to_c(type, 'arg%d' % i, 'x%d' % i, 'return NULL')
        prnt()
    prnt('  Py_BEGIN_ALLOW_THREADS')
    prnt('  _cffi_restore_errno();')
    call_arguments = ['x%d' % i for i in range(len(tp.args))]
    call_arguments = ', '.join(call_arguments)
    prnt('  { %s%s(%s); }' % (result_code, name, call_arguments))
    prnt('  _cffi_save_errno();')
    prnt('  Py_END_ALLOW_THREADS')
    prnt()
    prnt('  (void)self; /* unused */')
    if numargs == 0:
        prnt('  (void)noarg; /* unused */')
    if result_code:
        prnt('  pyresult = %s;' % self._convert_expr_from_c(tp.result, 'result', 'result type'))
        for freeline in freelines:
            prnt('  ' + freeline)
        prnt('  return pyresult;')
    else:
        for freeline in freelines:
            prnt('  ' + freeline)
        prnt('  Py_INCREF(Py_None);')
        prnt('  return Py_None;')
    prnt('}')
    prnt('#else')

    def need_indirection(type):
        return isinstance(type, model.StructOrUnion) or (isinstance(type, model.PrimitiveType) and type.is_complex_type())
    difference = False
    arguments = []
    call_arguments = []
    context = 'argument of %s' % name
    for i, type in enumerate(tp.args):
        indirection = ''
        if need_indirection(type):
            indirection = '*'
            difference = True
        arg = type.get_c_name(' %sx%d' % (indirection, i), context)
        arguments.append(arg)
        call_arguments.append('%sx%d' % (indirection, i))
    tp_result = tp.result
    if need_indirection(tp_result):
        context = 'result of %s' % name
        arg = tp_result.get_c_name(' *result', context)
        arguments.insert(0, arg)
        tp_result = model.void_type
        result_decl = None
        result_code = '*result = '
        difference = True
    if difference:
        repr_arguments = ', '.join(arguments)
        repr_arguments = repr_arguments or 'void'
        name_and_arguments = '%s_cffi_f_%s(%s)' % (abi, name, repr_arguments)
        prnt('static %s' % (tp_result.get_c_name(name_and_arguments),))
        prnt('{')
        if result_decl:
            prnt(result_decl)
        call_arguments = ', '.join(call_arguments)
        prnt('  { %s%s(%s); }' % (result_code, name, call_arguments))
        if result_decl:
            prnt('  return result;')
        prnt('}')
    else:
        prnt('#  define _cffi_f_%s _cffi_d_%s' % (name, name))
    prnt('#endif')
    prnt()