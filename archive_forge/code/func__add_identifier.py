from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def _add_identifier(self, name, coord):
    """ Add a new object, function, or enum member name (ie an ID) to the
            current scope
        """
    if self._scope_stack[-1].get(name, False):
        self._parse_error('Non-typedef %r previously declared as typedef in this scope' % name, coord)
    self._scope_stack[-1][name] = False