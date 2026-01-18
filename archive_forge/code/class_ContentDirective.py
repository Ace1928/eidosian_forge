import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class ContentDirective(Directive):
    """Implementation of the ``py:content`` template directive.
    
    This directive replaces the content of the element with the result of
    evaluating the value of the ``py:content`` attribute:
    
    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<ul xmlns:py="http://genshi.edgewall.org/">
    ...   <li py:content="bar">Hello</li>
    ... </ul>''')
    >>> print(tmpl.generate(bar='Bye'))
    <ul>
      <li>Bye</li>
    </ul>
    """
    __slots__ = []

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            raise TemplateSyntaxError('The content directive can not be used as an element', template.filepath, *pos[1:])
        expr = cls._parse_expr(value, template, *pos[1:])
        return (None, [stream[0], (EXPR, expr, pos), stream[-1]])