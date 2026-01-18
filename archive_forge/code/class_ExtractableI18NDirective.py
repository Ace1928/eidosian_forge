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
class ExtractableI18NDirective(I18NDirective):
    """Simple interface for directives to support messages extraction."""

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS, search_text=True, comment_stack=None, context_stack=None):
        raise NotImplementedError