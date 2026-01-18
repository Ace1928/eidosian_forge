import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class NamespaceTag(Tag):
    __keyword__ = 'namespace'

    def __init__(self, keyword, attributes, **kwargs):
        super().__init__(keyword, attributes, ('file',), ('name', 'inheritable', 'import', 'module'), (), **kwargs)
        self.name = attributes.get('name', '__anon_%s' % hex(abs(id(self))))
        if 'name' not in attributes and 'import' not in attributes:
            raise exceptions.CompileException("'name' and/or 'import' attributes are required for <%namespace>", **self.exception_kwargs)
        if 'file' in attributes and 'module' in attributes:
            raise exceptions.CompileException("<%namespace> may only have one of 'file' or 'module'", **self.exception_kwargs)

    def declared_identifiers(self):
        return []