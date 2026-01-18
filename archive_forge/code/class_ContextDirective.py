from gettext import NullTranslations
import os
import re
from functools import partial
from types import FunctionType
import six
from genshi.core import Attrs, Namespace, QName, START, END, TEXT, \
from genshi.template.base import DirectiveFactory, EXPR, SUB, _apply_directives
from genshi.template.directives import Directive, StripDirective
from genshi.template.markup import MarkupTemplate, EXEC
from genshi.compat import ast, IS_PYTHON2, _ast_Str, _ast_Str_value
class ContextDirective(I18NDirective):
    __slots__ = ['context']

    def __init__(self, value, template=None, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.context = value

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            value = value.get('name')
        return super(ContextDirective, cls).attach(template, stream, value, namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        ctxt.push({'_i18n.context': self.context})
        for event in _apply_directives(stream, directives, ctxt, vars):
            yield event
        ctxt.pop()