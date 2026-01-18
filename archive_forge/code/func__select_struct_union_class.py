from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def _select_struct_union_class(self, token):
    """ Given a token (either STRUCT or UNION), selects the
            appropriate AST class.
        """
    if token == 'struct':
        return c_ast.Struct
    else:
        return c_ast.Union