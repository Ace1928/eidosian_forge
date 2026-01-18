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
class RustCodeGen(CodeGen):
    """Generator for Rust code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.rs

    """
    code_extension = 'rs'

    def __init__(self, project='project', printer=None):
        super().__init__(project=project)
        self.printer = printer or RustCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Rust."""
        if is_sequence(expr) and (not isinstance(expr, (MatrixBase, MatrixExpr))):
            if not expr:
                raise ValueError('No expression given')
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)
        local_vars = {i.label for i in expressions.atoms(Idx)}
        global_vars = set() if global_vars is None else set(global_vars)
        symbols = expressions.free_symbols - local_vars - global_vars - expressions.atoms(Indexed)
        return_vals = []
        output_args = []
        for i, expr in enumerate(expressions):
            if isinstance(expr, Equality):
                out_arg = expr.lhs
                expr = expr.rhs
                symbol = out_arg
                if isinstance(out_arg, Indexed):
                    dims = tuple([(S.One, dim) for dim in out_arg.shape])
                    symbol = out_arg.base.label
                    output_args.append(InOutArgument(symbol, out_arg, expr, dimensions=dims))
                if not isinstance(out_arg, (Indexed, Symbol, MatrixSymbol)):
                    raise CodeGenError('Only Indexed, Symbol, or MatrixSymbol can define output arguments.')
                return_vals.append(Result(expr, name=symbol, result_var=out_arg))
                if not expr.has(symbol):
                    symbols.remove(symbol)
            else:
                return_vals.append(Result(expr, name='out%d' % (i + 1)))
        output_args.sort(key=lambda x: str(x.name))
        arg_list = list(output_args)
        array_symbols = {}
        for array in expressions.atoms(Indexed):
            array_symbols[array.base.label] = array
        for array in expressions.atoms(MatrixSymbol):
            array_symbols[array] = array
        for symbol in sorted(symbols, key=str):
            arg_list.append(InputArgument(symbol))
        if argument_sequence is not None:
            new_sequence = []
            for arg in argument_sequence:
                if isinstance(arg, IndexedBase):
                    new_sequence.append(arg.label)
                else:
                    new_sequence.append(arg)
            argument_sequence = new_sequence
            missing = [x for x in arg_list if x.name not in argument_sequence]
            if missing:
                msg = "Argument list didn't specify: {0} "
                msg = msg.format(', '.join([str(m.name) for m in missing]))
                raise CodeGenArgumentListError(msg, missing)
            name_arg_dict = {x.name: x for x in arg_list}
            new_args = []
            for symbol in argument_sequence:
                try:
                    new_args.append(name_arg_dict[symbol])
                except KeyError:
                    new_args.append(InputArgument(symbol))
            arg_list = new_args
        return Routine(name, arg_list, return_vals, local_vars, global_vars)

    def _get_header(self):
        """Writes a common header for the generated files."""
        code_lines = []
        code_lines.append('/*\n')
        tmp = header_comment % {'version': sympy_version, 'project': self.project}
        for line in tmp.splitlines():
            code_lines.append((' *%s' % line.center(76)).rstrip() + '\n')
        code_lines.append(' */\n')
        return code_lines

    def get_prototype(self, routine):
        """Returns a string for the function prototype of the routine.

        If the routine has multiple result objects, an CodeGenError is
        raised.

        See: https://en.wikipedia.org/wiki/Function_prototype

        """
        results = [i.get_datatype('Rust') for i in routine.results]
        if len(results) == 1:
            rstype = ' -> ' + results[0]
        elif len(routine.results) > 1:
            rstype = ' -> (' + ', '.join(results) + ')'
        else:
            rstype = ''
        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append(('*%s' % name, arg.get_datatype('Rust')))
            else:
                type_args.append((name, arg.get_datatype('Rust')))
        arguments = ', '.join(['%s: %s' % t for t in type_args])
        return 'fn %s(%s)%s' % (routine.name, arguments, rstype)

    def _preprocessor_statements(self, prefix):
        code_lines = []
        return code_lines

    def _get_routine_opening(self, routine):
        prototype = self.get_prototype(routine)
        return ['%s {\n' % prototype]

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        return []

    def _call_printer(self, routine):
        code_lines = []
        declarations = []
        returns = []
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and (not arg.dimensions):
                dereference.append(arg.name)
        for i, result in enumerate(routine.results):
            if isinstance(result, Result):
                assign_to = result.result_var
                returns.append(str(result.result_var))
            else:
                raise CodeGenError('unexpected object in Routine results')
            constants, not_supported, rs_expr = self._printer_method_with_settings('doprint', {'human': False}, result.expr, assign_to=assign_to)
            for name, value in sorted(constants, key=str):
                declarations.append('const %s: f64 = %s;\n' % (name, value))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append('// unsupported: %s\n' % name)
            code_lines.append('let %s\n' % rs_expr)
        if len(returns) > 1:
            returns = ['(' + ', '.join(returns) + ')']
        returns.append('\n')
        return declarations + code_lines + returns

    def _get_routine_ending(self, routine):
        return ['}\n']

    def dump_rs(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    dump_rs.extension = code_extension
    dump_rs.__doc__ = CodeGen.dump_code.__doc__
    dump_fns = [dump_rs]