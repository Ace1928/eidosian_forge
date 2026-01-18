from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def _is_type_in_scope(self, name):
    """ Is *name* a typedef-name in the current scope?
        """
    for scope in reversed(self._scope_stack):
        in_scope = scope.get(name)
        if in_scope is not None:
            return in_scope
    return False