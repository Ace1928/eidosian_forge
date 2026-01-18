from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
class SpecParser(object):
    """
    A parser that scans a file lurking for lines such as the one below.

    It then generates a pythran-compatible signature to inject into compile.
#pythran export a((float,(int, uint8),str list) list list)
#pythran export a(str)
#pythran export a( (str,str), int, int16 list list)
#pythran export a( {str} )
"""
    dtypes = {'bool': 'BOOL', 'byte': 'BYTE', 'complex': 'COMPLEX', 'int': 'INT', 'float': 'FLOAT', 'uint8': 'UINT8', 'uint16': 'UINT16', 'uint32': 'UINT32', 'uint64': 'UINT64', 'uintc': 'UINTC', 'uintp': 'UINTP', 'int8': 'INT8', 'int16': 'INT16', 'int32': 'INT32', 'int64': 'INT64', 'intc': 'INTC', 'intp': 'INTP', 'float32': 'FLOAT32', 'float64': 'FLOAT64', 'float128': 'FLOAT128', 'complex64': 'COMPLEX64', 'complex128': 'COMPLEX128', 'complex256': 'COMPLEX256'}
    reserved = {'#pythran': 'PYTHRAN', 'export': 'EXPORT', 'order': 'ORDER', 'capsule': 'CAPSULE', 'or': 'OR', 'list': 'LIST', 'set': 'SET', 'dict': 'DICT', 'slice': 'SLICE', 'str': 'STR', 'None': 'NONE'}
    reserved.update(dtypes)
    tokens = ('IDENTIFIER', 'NUM', 'COLUMN', 'LPAREN', 'RPAREN', 'CRAP', 'OPT', 'LARRAY', 'RARRAY', 'STAR', 'COMMA') + tuple(reserved.values())
    crap = [tok for tok in tokens if tok != 'PYTHRAN']
    some_crap = [tok for tok in crap if tok not in ('LPAREN', 'COMMA')]
    t_CRAP = '[^,:\\(\\)\\[\\]*?0-9]'
    t_COMMA = ','
    t_COLUMN = ':'
    t_LPAREN = '\\('
    t_RPAREN = '\\)'
    t_RARRAY = '\\]'
    t_LARRAY = '\\['
    t_STAR = '\\*'
    t_OPT = '\\?'
    t_NUM = '[1-9][0-9]*'
    precedence = (('left', 'OR'), ('left', 'LIST', 'DICT', 'SET'))

    def t_IDENTIFER(self, t):
        t.type = SpecParser.reserved.get(t.value, 'IDENTIFIER')
        return t
    t_IDENTIFER.__doc__ = '\\#?[a-zA-Z_][a-zA-Z_0-9]*'
    t_ignore = ' \t\n\r'

    def t_error(self, t):
        t.lexer.skip(1)

    def p_exports(self, p):
        if len(p) > 1:
            isnative = len(p) == 6
            target = self.exports if len(p) == 6 else self.native_exports
            for key, val in p[len(p) - 3]:
                target[key] += val
    p_exports.__doc__ = 'exports :\n                   | PYTHRAN EXPORT export_list opt_craps exports\n                   | PYTHRAN EXPORT CAPSULE export_list opt_craps exports'

    def p_export_list(self, p):
        p[0] = (p[1],) if len(p) == 2 else p[1] + (p[3],)
    p_export_list.__doc__ = 'export_list : export\n                  | export_list COMMA export'

    def p_export(self, p):
        if len(p) > 2:
            sigs = p[3] or ((),)
        else:
            sigs = ()
        p[0] = (p[1], sigs)
        self.export_info[p[1]] += (p.lexpos(1),)
    p_export.__doc__ = 'export : IDENTIFIER LPAREN opt_param_types RPAREN\n                  | IDENTIFIER\n                  | EXPORT LPAREN opt_param_types RPAREN\n                  | ORDER LPAREN opt_param_types RPAREN'

    def p_opt_craps(self, p):
        pass
    p_opt_craps.__doc__ = 'opt_craps :\n                     | some_crap opt_all_craps'

    def p_opt_all_craps(self, p):
        pass
    p_opt_all_craps.__doc__ = 'opt_all_craps :\n                     | crap opt_all_craps'

    def p_crap(self, p):
        pass
    p_crap.__doc__ = 'crap : ' + '\n| '.join(crap)

    def p_some_crap(self, p):
        pass
    p_some_crap.__doc__ = 'some_crap : ' + '\n| '.join(some_crap)

    def p_dtype(self, p):
        import numpy
        p[0] = (eval(p[1], numpy.__dict__),)
    p_dtype.__doc__ = 'dtype : ' + '\n| '.join(dtypes.values())

    def p_opt_param_types(self, p):
        p[0] = p[1] if len(p) == 2 else tuple()
    p_opt_param_types.__doc__ = 'opt_param_types :\n                     | param_types'

    def p_opt_types(self, p):
        p[0] = p[1] if len(p) == 2 else tuple()
    p_opt_types.__doc__ = 'opt_types :\n                     | types'

    def p_param_types(self, p):
        if len(p) == 2 or (len(p) == 3 and p[2] == ','):
            p[0] = tuple(((t,) for t in p[1]))
        elif len(p) == 3 and p[2] == '?':
            p[0] = tuple(((t,) for t in p[1])) + ((),)
        elif len(p) == 4:
            p[0] = tuple(((t,) + ts for t in p[1] for ts in p[3]))
        else:
            p[0] = tuple(((t,) + ts for t in p[1] for ts in p[4])) + ((),)
    p_param_types.__doc__ = 'param_types : type\n                       | type OPT\n                       | type COMMA\n                       | type COMMA param_types\n                       | type OPT COMMA default_types'

    def p_default_types(self, p):
        if len(p) == 3:
            p[0] = tuple(((t,) for t in p[1])) + ((),)
        else:
            p[0] = tuple(((t,) + ts for t in p[1] for ts in p[4])) + ((),)
    p_default_types.__doc__ = 'default_types : type OPT\n                       | type OPT COMMA default_types'

    def p_types(self, p):
        if len(p) == 2:
            p[0] = tuple(((t,) for t in p[1]))
        else:
            p[0] = tuple(((t,) + ts for t in p[1] for ts in p[3]))
    p_types.__doc__ = 'types : type\n                 | type COMMA types'

    def p_array_type(self, p):
        if len(p) == 2:
            p[0] = (p[1][0],)
        elif len(p) == 5 and p[4] == ']':

            def args(t):
                return t.__args__ if isinstance(t, NDArray) else (t,)
            p[0] = tuple((NDArray[args(t) + p[3]] for t in p[1]))
    p_array_type.__doc__ = 'array_type : dtype\n                | array_type LARRAY array_indices RARRAY'

    def p_type(self, p):
        if len(p) == 2:
            p[0] = (p[1],)
        elif len(p) == 3 and p[2] == 'list':
            p[0] = tuple((List[t] for t in p[1]))
        elif len(p) == 3 and p[2] == 'set':
            p[0] = tuple((Set[t] for t in p[1]))
        elif len(p) == 3:
            if p[2] is None:
                expanded = []
                for nd in p[1]:
                    expanded.append(nd)
                    if istransposable(nd):
                        expanded.append(NDArray[nd.__args__[0], -1:, -1:])
                p[0] = tuple(expanded)
            elif p[2] == 'F':
                for nd in p[1]:
                    if not istransposable(nd):
                        msg = 'Invalid Pythran spec. F order is only valid for 2D plain arrays'
                        self.p_error(p, msg)
                p[0] = tuple((NDArray[nd.__args__[0], -1:, -1:] for nd in p[1]))
            else:
                p[0] = p[1]
        elif len(p) == 5 and p[4] == ')':
            p[0] = tuple((Fun[args, r] for r in p[1] for args in (product(*p[3]) if len(p[3]) > 1 else p[3])))
        elif len(p) == 5:
            p[0] = tuple((Dict[k, v] for k in p[1] for v in p[3]))
        elif len(p) == 4 and p[2] == 'or':
            p[0] = p[1] + p[3]
        elif len(p) == 4 and p[3] == ')':
            p[0] = tuple((Tuple[t] for t in p[2]))
        elif len(p) == 4 and p[3] == ']':
            p[0] = p[2]
        else:
            msg = "Invalid Pythran spec. Unknown text '{0}'".format(p.value)
            self.p_error(p, msg)
    p_type.__doc__ = 'type : term\n                | array_type opt_order\n                | pointer_type\n                | type LIST\n                | type SET\n                | type LPAREN opt_types RPAREN\n                | type COLUMN type DICT\n                | LPAREN types RPAREN\n                | LARRAY type RARRAY\n                | type OR type\n                '

    def p_opt_order(self, p):
        if len(p) > 1:
            if p[3] not in 'CF':
                msg = "Invalid Pythran spec. Unknown order '{}'".format(p[3])
                self.p_error(p, msg)
            p[0] = p[3]
        else:
            p[0] = None
    p_opt_order.__doc__ = 'opt_order :\n                     | ORDER LPAREN IDENTIFIER RPAREN'

    def p_pointer_type(self, p):
        p[0] = Pointer[p[1][0]]
    p_pointer_type.__doc__ = 'pointer_type : dtype STAR'

    def p_array_indices(self, p):
        if len(p) == 2:
            p[0] = (p[1],)
        else:
            p[0] = (p[1],) + p[3]
    p_array_indices.__doc__ = 'array_indices : array_index\n                         | array_index COMMA array_indices'

    def p_array_index(self, p):
        if len(p) == 3:
            p[0] = slice(0, -1, -1)
        elif len(p) == 1 or p[1] == ':':
            p[0] = slice(0, -1, 1)
        else:
            p[0] = slice(0, int(p[1]), 1)
    p_array_index.__doc__ = 'array_index :\n                       | NUM\n                       | COLUMN\n                       | COLUMN COLUMN'

    def p_term(self, p):
        if p[1] == 'str':
            p[0] = str
        elif p[1] == 'slice':
            p[0] = slice
        elif p[1] == 'None':
            p[0] = type(None)
        else:
            p[0] = p[1][0]
    p_term.__doc__ = 'term : STR\n                | NONE\n                | SLICE\n                | dtype'

    def PythranSpecError(self, msg, lexpos=None):
        err = PythranSyntaxError(msg)
        if lexpos is not None:
            line_start = self.input_text.rfind('\n', 0, lexpos) + 1
            err.offset = lexpos - line_start
            err.lineno = 1 + self.input_text.count('\n', 0, lexpos)
        if self.input_file:
            err.filename = self.input_file
        return err

    def p_error(self, p):
        if p.type == 'IDENTIFIER':
            raise self.PythranSpecError('Unexpected identifier `{}` at that point'.format(p.value), p.lexpos)
        else:
            raise self.PythranSpecError('Unexpected token `{}` at that point'.format(p.value), p.lexpos)

    def __init__(self):
        self.lexer = lex.lex(module=self, debug=False)
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)

    def __call__(self, text, input_file=None):
        self.exports = defaultdict(tuple)
        self.native_exports = defaultdict(tuple)
        self.export_info = defaultdict(tuple)
        self.input_text = text
        self.input_file = input_file
        lines = []
        in_pythran_export = False
        for line in text.split('\n'):
            if re.match('\\s*#\\s*pythran', line):
                in_pythran_export = True
                lines.append(re.sub('\\s*#\\s*pythran', '#pythran', line))
            elif in_pythran_export:
                stripped = line.strip()
                if stripped.startswith('#'):
                    lines.append(line.replace('#', ''))
                else:
                    in_pythran_export = not stripped
                    lines.append('')
            else:
                in_pythran_export &= not line.strip()
                lines.append('')
        pythran_data = '\n'.join(lines)
        self.parser.parse(pythran_data, lexer=self.lexer, debug=False)
        for key, overloads in self.native_exports.items():
            if len(overloads) > 1:
                msg = "Overloads not supported for capsule '{}'".format(key)
                loc = self.export_info[key][-1]
                raise self.PythranSpecError(msg, loc)
            self.native_exports[key] = overloads[0]
        for key, overloads in self.exports.items():
            if len(overloads) > cfg.getint('typing', 'max_export_overloads'):
                raise self.PythranSpecError("Too many overloads for function '{}', probably due to automatic generation of C-style and Fortran-style memory layout. Please force a layout using `order(C)` or `order(F)` in the array signature".format(key))
            for i, ty_i in enumerate(overloads):
                sty_i = spec_to_string(key, ty_i)
                for ty_j in overloads[i + 1:]:
                    sty_j = spec_to_string(key, ty_j)
                    if sty_i == sty_j:
                        msg = 'Duplicate export entry {}.'.format(sty_i)
                        loc = self.export_info[key][-1]
                        raise self.PythranSpecError(msg, loc)
                    if ambiguous_types(ty_i, ty_j):
                        msg = 'Ambiguous overloads\n\t{}\n\t{}.'.format(sty_i, sty_j)
                        loc = self.export_info[key][i]
                        raise self.PythranSpecError(msg, loc)
        return Spec(self.exports, self.native_exports)