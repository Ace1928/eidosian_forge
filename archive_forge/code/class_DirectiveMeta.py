import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class DirectiveMeta(type):
    """Meta class for template directives."""

    def __new__(cls, name, bases, d):
        d['tagname'] = name.lower().replace('directive', '')
        return type.__new__(cls, name, bases, d)