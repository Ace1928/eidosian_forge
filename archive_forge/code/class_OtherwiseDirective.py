import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class OtherwiseDirective(Directive):
    """Implementation of the ``py:otherwise`` directive for nesting in a parent
    with the ``py:choose`` directive.
    
    See the documentation of `ChooseDirective` for usage.
    """
    __slots__ = ['filename']

    def __init__(self, value, template, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.filename = template.filepath

    def __call__(self, stream, directives, ctxt, **vars):
        info = ctxt._choice_stack and ctxt._choice_stack[-1]
        if not info:
            raise TemplateRuntimeError('an "otherwise" directive can only be used inside a "choose" directive', self.filename, *next(stream)[2][1:])
        if info[0]:
            return []
        info[0] = True
        return _apply_directives(stream, directives, ctxt, vars)