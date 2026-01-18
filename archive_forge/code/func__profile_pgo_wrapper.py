from __future__ import absolute_import, print_function
import io
import os
import re
import sys
import time
import copy
import distutils.log
import textwrap
import hashlib
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from IPython.core import display
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.utils.text import dedent
from ..Shadow import __version__ as cython_version
from ..Compiler.Errors import CompileError
from .Inline import cython_inline, load_dynamic
from .Dependencies import cythonize
from ..Utils import captured_fd, print_captured
def _profile_pgo_wrapper(self, extension, lib_dir):
    """
        Generate a .c file for a separate extension module that calls the
        module init function of the original module.  This makes sure that the
        PGO profiler sees the correct .o file of the final module, but it still
        allows us to import the module under a different name for profiling,
        before recompiling it into the PGO optimised module.  Overwriting and
        reimporting the same shared library is not portable.
        """
    extension = copy.copy(extension)
    module_name = extension.name
    pgo_module_name = '_pgo_' + module_name
    pgo_wrapper_c_file = os.path.join(lib_dir, pgo_module_name + '.c')
    with io.open(pgo_wrapper_c_file, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent(u'\n            #include "Python.h"\n            #if PY_MAJOR_VERSION < 3\n            extern PyMODINIT_FUNC init%(module_name)s(void);\n            PyMODINIT_FUNC init%(pgo_module_name)s(void); /*proto*/\n            PyMODINIT_FUNC init%(pgo_module_name)s(void) {\n                PyObject *sys_modules;\n                init%(module_name)s();  if (PyErr_Occurred()) return;\n                sys_modules = PyImport_GetModuleDict();  /* borrowed, no exception, "never" fails */\n                if (sys_modules) {\n                    PyObject *module = PyDict_GetItemString(sys_modules, "%(module_name)s");  if (!module) return;\n                    PyDict_SetItemString(sys_modules, "%(pgo_module_name)s", module);\n                    Py_DECREF(module);\n                }\n            }\n            #else\n            extern PyMODINIT_FUNC PyInit_%(module_name)s(void);\n            PyMODINIT_FUNC PyInit_%(pgo_module_name)s(void); /*proto*/\n            PyMODINIT_FUNC PyInit_%(pgo_module_name)s(void) {\n                return PyInit_%(module_name)s();\n            }\n            #endif\n            ' % {'module_name': module_name, 'pgo_module_name': pgo_module_name}))
    extension.sources = extension.sources + [pgo_wrapper_c_file]
    extension.name = pgo_module_name
    self._build_extension(extension, lib_dir, pgo_step_name='gen')
    so_module_path = os.path.join(lib_dir, pgo_module_name + self.so_ext)
    load_dynamic(pgo_module_name, so_module_path)