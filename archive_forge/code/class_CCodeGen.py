import os
import textwrap
from io import StringIO
from sympy import __version__ as sympy_version
from sympy.core import Symbol, S, Tuple, Equality, Function, Basic
from sympy.printing.c import c_code_printers
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.fortran import FCodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.rust import RustCodePrinter
from sympy.tensor import Idx, Indexed, IndexedBase
from sympy.matrices import (MatrixSymbol, ImmutableMatrix, MatrixBase,
from sympy.utilities.iterables import is_sequence
class CCodeGen(CodeGen):
    """Generator for C code.

    The .write() method inherited from CodeGen will output a code file and
    an interface file, <prefix>.c and <prefix>.h respectively.

    """
    code_extension = 'c'
    interface_extension = 'h'
    standard = 'c99'

    def __init__(self, project='project', printer=None, preprocessor_statements=None, cse=False):
        super().__init__(project=project, cse=cse)
        self.printer = printer or c_code_printers[self.standard.lower()]()
        self.preprocessor_statements = preprocessor_statements
        if preprocessor_statements is None:
            self.preprocessor_statements = ['#include <math.h>']

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append('/' + '*' * 78 + '\n')
        tmp = header_comment % {'version': sympy_version, 'project': self.project}
        for line in tmp.splitlines():
            code_lines.append(' *%s*\n' % line.center(76))
        code_lines.append(' ' + '*' * 78 + '/\n')
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        if len(routine.results) > 1:
            raise CodeGenError('C only supports a single or no return value.')
        elif len(routine.results) == 1:
            ctype = routine.results[0].get_datatype('C')
        else:
            ctype = 'void'
        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append((arg.get_datatype('C'), '*%s' % name))
            else:
                type_args.append((arg.get_datatype('C'), name))
        arguments = ', '.join(['%s %s' % t for t in type_args])
        return '%s %s(%s)' % (ctype, routine.name, arguments)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        code_lines.append('#include "{}.h"'.format(os.path.basename(prefix)))
        code_lines.extend(self.preprocessor_statements)
        code_lines = ['{}\n'.format(l) for l in code_lines]
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ['%s {\n' % prototype]

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and (not arg.dimensions):
                dereference.append(arg.name)
        code_lines = []
        for result in routine.local_vars:
            if not isinstance(result, Result):
                continue
            if result.name != result.result_var:
                raise CodeGen('Result variable and name should match: {}'.format(result))
            assign_to = result.name
            t = result.get_datatype('c')
            if isinstance(result.expr, (MatrixBase, MatrixExpr)):
                dims = result.expr.shape
                code_lines.append('{} {}[{}];\n'.format(t, str(assign_to), dims[0] * dims[1]))
                prefix = ''
            else:
                prefix = 'const {} '.format(t)
            constants, not_c, c_expr = self._printer_method_with_settings('doprint', {'human': False, 'dereference': dereference}, result.expr, assign_to=assign_to)
            for name, value in sorted(constants, key=str):
                code_lines.append('double const %s = %s;\n' % (name, value))
            code_lines.append('{}{}\n'.format(prefix, c_expr))
        return code_lines

    def _call_printer(self, routine):
        code_lines = []
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and (not arg.dimensions):
                dereference.append(arg.name)
        return_val = None
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name + '_result'
                t = result.get_datatype('c')
                code_lines.append('{} {};\n'.format(t, str(assign_to)))
                return_val = assign_to
            else:
                assign_to = result.result_var
            try:
                constants, not_c, c_expr = self._printer_method_with_settings('doprint', {'human': False, 'dereference': dereference}, result.expr, assign_to=assign_to)
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append('%s %s;\n' % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = self._printer_method_with_settings('doprint', {'human': False, 'dereference': dereference}, result.expr, assign_to=assign_to)
            for name, value in sorted(constants, key=str):
                code_lines.append('double const %s = %s;\n' % (name, value))
            code_lines.append('%s\n' % c_expr)
        if return_val:
            code_lines.append('   return %s;\n' % return_val)
        return code_lines

    def _get_routine_ending(self, routine):
        return ['}\n']

    def dump_c(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    dump_c.extension = code_extension
    dump_c.__doc__ = CodeGen.dump_code.__doc__

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        """Writes the C header file.

        This file contains all the function declarations.

        Parameters
        ==========

        routines : list
            A list of Routine instances.

        f : file-like
            Where to write the file.

        prefix : string
            The filename prefix, used to construct the include guards.
            Only the basename of the prefix is used.

        header : bool, optional
            When True, a header comment is included on top of each source
            file.  [default : True]

        empty : bool, optional
            When True, empty lines are included to structure the source
            files.  [default : True]

        """
        if header:
            print(''.join(self._get_header()), file=f)
        guard_name = '%s__%s__H' % (self.project.replace(' ', '_').upper(), prefix.replace('/', '_').upper())
        if empty:
            print(file=f)
        print('#ifndef %s' % guard_name, file=f)
        print('#define %s' % guard_name, file=f)
        if empty:
            print(file=f)
        for routine in routines:
            prototype = self.get_prototype(routine)
            print('%s;' % prototype, file=f)
        if empty:
            print(file=f)
        print('#endif', file=f)
        if empty:
            print(file=f)
    dump_h.extension = interface_extension
    dump_fns = [dump_c, dump_h]