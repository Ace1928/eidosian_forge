import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class ForDirective(Directive):
    """Implementation of the ``py:for`` template directive for repeating an
    element based on an iterable in the context data.
    
    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<ul xmlns:py="http://genshi.edgewall.org/">
    ...   <li py:for="item in items">${item}</li>
    ... </ul>''')
    >>> print(tmpl.generate(items=[1, 2, 3]))
    <ul>
      <li>1</li><li>2</li><li>3</li>
    </ul>
    """
    __slots__ = ['assign', 'filename']

    def __init__(self, value, template, namespaces=None, lineno=-1, offset=-1):
        if ' in ' not in value:
            raise TemplateSyntaxError('"in" keyword missing in "for" directive', template.filepath, lineno, offset)
        assign, value = value.split(' in ', 1)
        ast = _parse(assign, 'exec')
        value = 'iter(%s)' % value.strip()
        self.assign = _assignment(ast.body[0].value)
        self.filename = template.filepath
        Directive.__init__(self, value, template, namespaces, lineno, offset)

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            value = value.get('each')
        return super(ForDirective, cls).attach(template, stream, value, namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        iterable = _eval_expr(self.expr, ctxt, vars)
        if iterable is None:
            return
        assign = self.assign
        scope = {}
        stream = list(stream)
        for item in iterable:
            assign(scope, item)
            ctxt.push(scope)
            for event in _apply_directives(stream, directives, ctxt, vars):
                yield event
            ctxt.pop()

    def __repr__(self):
        return '<%s>' % type(self).__name__