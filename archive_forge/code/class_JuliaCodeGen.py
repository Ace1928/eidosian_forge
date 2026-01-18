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
class JuliaCodeGen(CodeGen):
    """Generator for Julia code.

    The .write() method inherited from CodeGen will output a code file
    <prefix>.jl.

    """
    code_extension = 'jl'

    def __init__(self, project='project', printer=None):
        super().__init__(project)
        self.printer = printer or JuliaCodePrinter()

    def routine(self, name, expr, argument_sequence, global_vars):
        """Specialized Routine creation for Julia."""
        if is_sequence(expr) and (not isinstance(expr, (MatrixBase, MatrixExpr))):
            if not expr:
                raise ValueError('No expression given')
            expressions = Tuple(*expr)
        else:
            expressions = Tuple(expr)
        local_vars = {i.label for i in expressions.atoms(Idx)}
        global_vars = set() if global_vars is None else set(global_vars)
        old_symbols = expressions.free_symbols - local_vars - global_vars
        symbols = set()
        for s in old_symbols:
            if isinstance(s, Idx):
                symbols.update(s.args[1].free_symbols)
            elif not isinstance(s, Indexed):
                symbols.add(s)
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
        tmp = header_comment % {'version': sympy_version, 'project': self.project}
        for line in tmp.splitlines():
            if line == '':
                code_lines.append('#\n')
            else:
                code_lines.append('#   %s\n' % line)
        return code_lines

    def _preprocessor_statements(self, prefix):
        return []

    def _get_routine_opening(self, routine):
        """Returns the opening statements of the routine."""
        code_list = []
        code_list.append('function ')
        args = []
        for i, arg in enumerate(routine.arguments):
            if isinstance(arg, OutputArgument):
                raise CodeGenError('Julia: invalid argument of type %s' % str(type(arg)))
            if isinstance(arg, (InputArgument, InOutArgument)):
                args.append('%s' % self._get_symbol(arg.name))
        args = ', '.join(args)
        code_list.append('%s(%s)\n' % (routine.name, args))
        code_list = [''.join(code_list)]
        return code_list

    def _declare_arguments(self, routine):
        return []

    def _declare_globals(self, routine):
        return []

    def _declare_locals(self, routine):
        return []

    def _get_routine_ending(self, routine):
        outs = []
        for result in routine.results:
            if isinstance(result, Result):
                s = self._get_symbol(result.name)
            else:
                raise CodeGenError('unexpected object in Routine results')
            outs.append(s)
        return ['return ' + ', '.join(outs) + '\nend\n']

    def _call_printer(self, routine):
        declarations = []
        code_lines = []
        for i, result in enumerate(routine.results):
            if isinstance(result, Result):
                assign_to = result.result_var
            else:
                raise CodeGenError('unexpected object in Routine results')
            constants, not_supported, jl_expr = self._printer_method_with_settings('doprint', {'human': False}, result.expr, assign_to=assign_to)
            for obj, v in sorted(constants, key=str):
                declarations.append('%s = %s\n' % (obj, v))
            for obj in sorted(not_supported, key=str):
                if isinstance(obj, Function):
                    name = obj.func
                else:
                    name = obj
                declarations.append('# unsupported: %s\n' % name)
            code_lines.append('%s\n' % jl_expr)
        return declarations + code_lines

    def _indent_code(self, codelines):
        p = JuliaCodePrinter({'human': False})
        return p.indent_code(codelines)

    def dump_jl(self, routines, f, prefix, header=True, empty=True):
        self.dump_code(routines, f, prefix, header, empty)
    dump_jl.extension = code_extension
    dump_jl.__doc__ = CodeGen.dump_code.__doc__
    dump_fns = [dump_jl]