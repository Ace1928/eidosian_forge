import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class StripDirective(Directive):
    """Implementation of the ``py:strip`` template directive.
    
    When the value of the ``py:strip`` attribute evaluates to ``True``, the
    element is stripped from the output
    
    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <div py:strip="True"><b>foo</b></div>
    ... </div>''')
    >>> print(tmpl.generate())
    <div>
      <b>foo</b>
    </div>
    
    Leaving the attribute value empty is equivalent to a truth value.
    
    This directive is particulary interesting for named template functions or
    match templates that do not generate a top-level element:
    
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <div py:def="echo(what)" py:strip="">
    ...     <b>${what}</b>
    ...   </div>
    ...   ${echo('foo')}
    ... </div>''')
    >>> print(tmpl.generate())
    <div>
        <b>foo</b>
    </div>
    """
    __slots__ = []

    def __call__(self, stream, directives, ctxt, **vars):

        def _generate():
            if not self.expr or _eval_expr(self.expr, ctxt, vars):
                next(stream)
                previous = next(stream)
                for event in stream:
                    yield previous
                    previous = event
            else:
                for event in stream:
                    yield event
        return _apply_directives(_generate(), directives, ctxt, vars)