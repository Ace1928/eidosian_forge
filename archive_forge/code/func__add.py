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
def _add(arg):
    if isinstance(arg, _ast_Str) and isinstance(_ast_Str_value(arg), six.text_type):
        strings.append(_ast_Str_value(arg))
    elif isinstance(arg, _ast_Str):
        strings.append(six.text_type(_ast_Str_value(arg), 'utf-8'))
    elif arg:
        strings.append(None)