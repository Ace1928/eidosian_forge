from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def add_function_to(self, to, func, name, ctypes, signature):
    """
        Add a function to be exposed. *func* is expected to be a
        :class:`cgen.FunctionBody`.

        Because a function can have several signatures exported,
        this method actually creates a wrapper for each specialization
        and a global wrapper that checks the argument types and
        runs the correct candidate, if any
        """
    to.append(func)
    args_unboxing = []
    args_checks = []
    wrapper_name = pythran_ward + 'wrap_' + func.fdecl.name
    for i, t in enumerate(ctypes):
        args_unboxing.append('from_python<{}>(args_obj[{}])'.format(t, i))
        args_checks.append('is_convertible<{}>(args_obj[{}])'.format(t, i))
    arg_decls = func.fdecl.arg_decls[:len(ctypes)]
    keywords = ''.join(('"{}", '.format(arg.name) for arg in arg_decls))
    wrapper = dedent('\n            static PyObject *\n            {wname}(PyObject *self, PyObject *args, PyObject *kw)\n            {{\n                PyObject* args_obj[{size}+1];\n                {silent_warning}\n                char const* keywords[] = {{{keywords} nullptr}};\n                if(! PyArg_ParseTupleAndKeywords(args, kw, "{fmt}",\n                                                 (char**)keywords {objs}))\n                    return nullptr;\n                if({checks})\n                    return to_python({name}({args}));\n                else {{\n                    return nullptr;\n                }}\n            }}')
    self.wrappers.append(wrapper.format(name=func.fdecl.name, silent_warning='' if ctypes else '(void)args_obj;', size=len(ctypes), fmt='O' * len(ctypes), objs=''.join((', &args_obj[%d]' % i for i in range(len(ctypes)))), args=', '.join(args_unboxing), checks=' && '.join(args_checks) or '1', wname=wrapper_name, keywords=keywords))
    func_descriptor = (wrapper_name, ctypes, signature)
    self.functions.setdefault(name, []).append(func_descriptor)