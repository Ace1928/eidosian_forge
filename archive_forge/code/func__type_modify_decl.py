from .ply import yacc
from . import c_ast
from .c_lexer import CLexer
from .plyparser import PLYParser, ParseError, parameterized, template
from .ast_transforms import fix_switch_cases, fix_atomic_specifiers
def _type_modify_decl(self, decl, modifier):
    """ Tacks a type modifier on a declarator, and returns
            the modified declarator.

            Note: the declarator and modifier may be modified
        """
    modifier_head = modifier
    modifier_tail = modifier
    while modifier_tail.type:
        modifier_tail = modifier_tail.type
    if isinstance(decl, c_ast.TypeDecl):
        modifier_tail.type = decl
        return modifier
    else:
        decl_tail = decl
        while not isinstance(decl_tail.type, c_ast.TypeDecl):
            decl_tail = decl_tail.type
        modifier_tail.type = decl_tail.type
        decl_tail.type = modifier_head
        return decl