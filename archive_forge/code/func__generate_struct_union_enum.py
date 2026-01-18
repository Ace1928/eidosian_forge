from . import c_ast
def _generate_struct_union_enum(self, n, name):
    """ Generates code for structs, unions, and enums. name should be
            'struct', 'union', or 'enum'.
        """
    if name in ('struct', 'union'):
        members = n.decls
        body_function = self._generate_struct_union_body
    else:
        assert name == 'enum'
        members = None if n.values is None else n.values.enumerators
        body_function = self._generate_enum_body
    s = name + ' ' + (n.name or '')
    if members is not None:
        s += '\n'
        s += self._make_indent()
        self.indent_level += 2
        s += '{\n'
        s += body_function(members)
        self.indent_level -= 2
        s += self._make_indent() + '}'
    return s