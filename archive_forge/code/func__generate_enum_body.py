from . import c_ast
def _generate_enum_body(self, members):
    return ''.join((self.visit(value) for value in members))[:-2] + '\n'