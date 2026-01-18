import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class WithDirective(Directive):
    """Implementation of the ``py:with`` template directive, which allows
    shorthand access to variables and expressions.
    
    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <span py:with="y=7; z=x+10">$x $y $z</span>
    ... </div>''')
    >>> print(tmpl.generate(x=42))
    <div>
      <span>42 7 52</span>
    </div>
    """
    __slots__ = ['vars']

    def __init__(self, value, template, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.vars = []
        value = value.strip()
        try:
            ast = _parse(value, 'exec')
            for node in ast.body:
                if not isinstance(node, _ast.Assign):
                    raise TemplateSyntaxError('only assignment allowed in value of the "with" directive', template.filepath, lineno, offset)
                self.vars.append(([_assignment(n) for n in node.targets], Expression(node.value, template.filepath, lineno, lookup=template.lookup)))
        except SyntaxError as err:
            err.msg += ' in expression "%s" of "%s" directive' % (value, self.tagname)
            raise TemplateSyntaxError(err, template.filepath, lineno, offset + (err.offset or 0))

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            value = value.get('vars')
        return super(WithDirective, cls).attach(template, stream, value, namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        frame = {}
        ctxt.push(frame)
        for targets, expr in self.vars:
            value = _eval_expr(expr, ctxt, vars)
            for assign in targets:
                assign(frame, value)
        for event in _apply_directives(stream, directives, ctxt, vars):
            yield event
        ctxt.pop()

    def __repr__(self):
        return '<%s>' % type(self).__name__