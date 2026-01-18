from . import c_ast
def _generate_struct_union_body(self, members):
    return ''.join((self._generate_stmt(decl) for decl in members))