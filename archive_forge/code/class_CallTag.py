import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class CallTag(Tag):
    __keyword__ = 'call'

    def __init__(self, keyword, attributes, **kwargs):
        super().__init__(keyword, attributes, 'args', ('expr',), ('expr',), **kwargs)
        self.expression = attributes['expr']
        self.code = ast.PythonCode(self.expression, **self.exception_kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        return self.code.declared_identifiers.union(self.body_decl.allargnames)

    def undeclared_identifiers(self):
        return self.code.undeclared_identifiers.difference(self.code.declared_identifiers)