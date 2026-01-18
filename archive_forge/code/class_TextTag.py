import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class TextTag(Tag):
    __keyword__ = 'text'

    def __init__(self, keyword, attributes, **kwargs):
        super().__init__(keyword, attributes, (), 'filter', (), **kwargs)
        self.filter_args = ast.ArgumentList(attributes.get('filter', ''), **self.exception_kwargs)

    def undeclared_identifiers(self):
        return self.filter_args.undeclared_identifiers.difference(filters.DEFAULT_ESCAPES.keys()).union(self.expression_undeclared_identifiers)