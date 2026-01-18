import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class AttrsDirective(Directive):
    """Implementation of the ``py:attrs`` template directive.
    
    The value of the ``py:attrs`` attribute should be a dictionary or a sequence
    of ``(name, value)`` tuples. The items in that dictionary or sequence are
    added as attributes to the element:
    
    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<ul xmlns:py="http://genshi.edgewall.org/">
    ...   <li py:attrs="foo">Bar</li>
    ... </ul>''')
    >>> print(tmpl.generate(foo={'class': 'collapse'}))
    <ul>
      <li class="collapse">Bar</li>
    </ul>
    >>> print(tmpl.generate(foo=[('class', 'collapse')]))
    <ul>
      <li class="collapse">Bar</li>
    </ul>
    
    If the value evaluates to ``None`` (or any other non-truth value), no
    attributes are added:
    
    >>> print(tmpl.generate(foo=None))
    <ul>
      <li>Bar</li>
    </ul>
    """
    __slots__ = []

    def __call__(self, stream, directives, ctxt, **vars):

        def _generate():
            kind, (tag, attrib), pos = next(stream)
            attrs = _eval_expr(self.expr, ctxt, vars)
            if attrs:
                if isinstance(attrs, Stream):
                    try:
                        attrs = next(iter(attrs))
                    except StopIteration:
                        attrs = []
                elif not isinstance(attrs, list):
                    attrs = attrs.items()
                attrib |= [(QName(n), v is not None and six.text_type(v).strip() or None) for n, v in attrs]
            yield (kind, (tag, attrib), pos)
            for event in stream:
                yield event
        return _apply_directives(_generate(), directives, ctxt, vars)