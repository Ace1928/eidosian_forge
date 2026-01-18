import tempfile
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip
from sympy.codegen.ast import (
from sympy.codegen.cnodes import void, PreIncrement
from sympy.codegen.cutils import render_as_source_file
def _render_compile_import(funcdef, build_dir):
    code_str = render_as_source_file(funcdef, settings={'contract': False})
    declar = ccode(FunctionPrototype.from_FunctionDefinition(funcdef))
    return compile_link_import_strings([('our_test_func.c', code_str), ('_our_test_func.pyx', ('#cython: language_level={}\n'.format('3') + 'cdef extern {declar}\ndef _{fname}({typ}[:] inp, {typ}[:] out):\n    {fname}(inp.size, &inp[0], &out[0])').format(declar=declar, fname=funcdef.name, typ='double'))], build_dir=build_dir)