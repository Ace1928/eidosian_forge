from . import c_ast
def _generate_type(self, n, modifiers=[], emit_declname=True):
    """ Recursive generation from a type node. n is the type node.
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
    typ = type(n)
    if typ == c_ast.TypeDecl:
        s = ''
        if n.quals:
            s += ' '.join(n.quals) + ' '
        s += self.visit(n.type)
        nstr = n.declname if n.declname and emit_declname else ''
        for i, modifier in enumerate(modifiers):
            if isinstance(modifier, c_ast.ArrayDecl):
                if i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl):
                    nstr = '(' + nstr + ')'
                nstr += '['
                if modifier.dim_quals:
                    nstr += ' '.join(modifier.dim_quals) + ' '
                nstr += self.visit(modifier.dim) + ']'
            elif isinstance(modifier, c_ast.FuncDecl):
                if i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl):
                    nstr = '(' + nstr + ')'
                nstr += '(' + self.visit(modifier.args) + ')'
            elif isinstance(modifier, c_ast.PtrDecl):
                if modifier.quals:
                    nstr = '* %s%s' % (' '.join(modifier.quals), ' ' + nstr if nstr else '')
                else:
                    nstr = '*' + nstr
        if nstr:
            s += ' ' + nstr
        return s
    elif typ == c_ast.Decl:
        return self._generate_decl(n.type)
    elif typ == c_ast.Typename:
        return self._generate_type(n.type, emit_declname=emit_declname)
    elif typ == c_ast.IdentifierType:
        return ' '.join(n.names) + ' '
    elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
        return self._generate_type(n.type, modifiers + [n], emit_declname=emit_declname)
    else:
        return self.visit(n)