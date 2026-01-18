import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class PageTag(Tag):
    __keyword__ = 'page'

    def __init__(self, keyword, attributes, **kwargs):
        expressions = ['cached', 'args', 'expression_filter', 'enable_loop'] + [c for c in attributes if c.startswith('cache_')]
        super().__init__(keyword, attributes, expressions, (), (), **kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)
        self.filter_args = ast.ArgumentList(attributes.get('expression_filter', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        return self.body_decl.allargnames