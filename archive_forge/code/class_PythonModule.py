from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class PythonModule(object):
    """
    Wraps the creation of a Pythran module wrapped a Python native Module
    """

    def __init__(self, name, docstrings, metadata):
        """
        Builds an empty PythonModule
        """
        self.name = name
        self.preamble = []
        self.includes = []
        self.functions = {}
        self.global_vars = []
        self.implems = []
        self.capsules = []
        self.python_implems = []
        self.wrappers = []
        self.docstrings = docstrings
        self.metadata = metadata
        moduledoc = self.docstring(self.docstrings.get(None, ''))
        self.metadata['moduledoc'] = moduledoc

    def docstring(self, doc):
        return self.splitstring(quote_cxxstring(dedent(doc)))

    def splitstring(self, doc):
        return '"{}"'.format('\\n""'.join(doc.split('\\n')))

    def add_to_preamble(self, *pa):
        self.preamble.extend(pa)

    def add_to_includes(self, *incl):
        self.includes.extend(incl)

    def add_pyfunction(self, func, name, types, signature):
        self.add_function_to(self.python_implems, func, name, types, signature)

    def add_capsule(self, func, ptrname, sig):
        self.capsules.append((ptrname, sig))
        self.implems.append(func)

    def add_function(self, func, name, types, signature):
        self.add_function_to(self.implems, func, name, types, signature)

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

    def add_global_var(self, name, init):
        self.global_vars.append(name)
        self.python_implems.append(Assign('static PyObject* ' + name, 'to_python({})'.format(init)))

    def __str__(self):
        """Generate (i.e. yield) the source code of the
        module line-by-line.
        """
        themethods = []
        theextraobjects = []
        theoverloads = []
        for vname in self.global_vars:
            theextraobjects.append('PyModule_AddObject(theModule, "{0}", {0});'.format(vname))
        for fname, overloads in self.functions.items():
            tryall = []
            signatures = []
            for overload, ctypes, signature in overloads:
                try_ = dedent('\n                    if(PyObject* obj = {name}(self, args, kw))\n                        return obj;\n                    PyErr_Clear();\n                    '.format(name=overload))
                tryall.append(try_)
                signatures.append(signature)
            candidates = signatures_to_string(fname, signatures)
            wrapper_name = pythran_ward + 'wrapall_' + fname
            candidate = dedent('\n            static PyObject *\n            {wname}(PyObject *self, PyObject *args, PyObject *kw)\n            {{\n                return pythonic::handle_python_exception([self, args, kw]()\n                -> PyObject* {{\n                {tryall}\n                return pythonic::python::raise_invalid_argument(\n                               "{name}", {candidates}, args, kw);\n                }});\n            }}\n            '.format(name=fname, tryall='\n'.join(tryall), candidates=self.splitstring(candidates.replace('\n', '\\n')), wname=wrapper_name))
            fdoc = self.docstring(self.docstrings.get(fname, ''))
            themethod = dedent('{{\n                "{name}",\n                (PyCFunction){wname},\n                METH_VARARGS | METH_KEYWORDS,\n                {doc}}}'.format(name=fname, wname=wrapper_name, doc=fdoc))
            themethods.append(themethod)
            theoverloads.append(candidate)
        for ptrname, sig in self.capsules:
            capsule = '\n            PyModule_AddObject(theModule, "{ptrname}",\n                               PyCapsule_New((void*)&{ptrname}, "{sig}", NULL)\n            );'.format(ptrname=ptrname, sig=sig)
            theextraobjects.append(capsule)
        methods = dedent('\n            static PyMethodDef Methods[] = {{\n                {methods}\n                {{NULL, NULL, 0, NULL}}\n            }};\n            '.format(methods=''.join((m + ',' for m in themethods))))
        module = dedent('\n            #if PY_MAJOR_VERSION >= 3\n              static struct PyModuleDef moduledef = {{\n                PyModuleDef_HEAD_INIT,\n                "{name}",            /* m_name */\n                {moduledoc},         /* m_doc */\n                -1,                  /* m_size */\n                Methods,             /* m_methods */\n                NULL,                /* m_reload */\n                NULL,                /* m_traverse */\n                NULL,                /* m_clear */\n                NULL,                /* m_free */\n              }};\n            #define PYTHRAN_RETURN return theModule\n            #define PYTHRAN_MODULE_INIT(s) PyInit_##s\n            #else\n            #define PYTHRAN_RETURN return\n            #define PYTHRAN_MODULE_INIT(s) init##s\n            #endif\n            PyMODINIT_FUNC\n            PYTHRAN_MODULE_INIT({name})(void)\n            #ifndef _WIN32\n            __attribute__ ((visibility("default")))\n            #if defined(GNUC) && !defined(__clang__)\n            __attribute__ ((externally_visible))\n            #endif\n            #endif\n            ;\n            PyMODINIT_FUNC\n            PYTHRAN_MODULE_INIT({name})(void) {{\n                import_array()\n                #if PY_MAJOR_VERSION >= 3\n                PyObject* theModule = PyModule_Create(&moduledef);\n                #else\n                PyObject* theModule = Py_InitModule3("{name}",\n                                                     Methods,\n                                                     {moduledoc}\n                );\n                #endif\n                if(! theModule)\n                    PYTHRAN_RETURN;\n                PyObject * theDoc = Py_BuildValue("(ss)",\n                                                  "{version}",\n                                                  "{hash}");\n                if(! theDoc)\n                    PYTHRAN_RETURN;\n                PyModule_AddObject(theModule,\n                                   "__pythran__",\n                                   theDoc);\n\n                {extraobjects}\n                PYTHRAN_RETURN;\n            }}\n            '.format(name=self.name, extraobjects='\n'.join(theextraobjects), **self.metadata))
        body = self.preamble + self.includes + self.implems + [Line('#ifdef ENABLE_PYTHON_MODULE')] + self.python_implems + [Line(code) for code in self.wrappers + theoverloads] + [Line(methods), Line(module), Line('#endif')]
        return '\n'.join(Module(body).generate())