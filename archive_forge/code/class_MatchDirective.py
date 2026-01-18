import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class MatchDirective(Directive):
    """Implementation of the ``py:match`` template directive.

    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <span py:match="greeting">
    ...     Hello ${select('@name')}
    ...   </span>
    ...   <greeting name="Dude" />
    ... </div>''')
    >>> print(tmpl.generate())
    <div>
      <span>
        Hello Dude
      </span>
    </div>
    """
    __slots__ = ['path', 'namespaces', 'hints']

    def __init__(self, value, template, hints=None, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.path = Path(value, template.filepath, lineno)
        self.namespaces = namespaces or {}
        self.hints = hints or ()

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        hints = []
        if type(value) is dict:
            if value.get('buffer', '').lower() == 'false':
                hints.append('not_buffered')
            if value.get('once', '').lower() == 'true':
                hints.append('match_once')
            if value.get('recursive', '').lower() == 'false':
                hints.append('not_recursive')
            value = value.get('path')
        return (cls(value, template, frozenset(hints), namespaces, *pos[1:]), stream)

    def __call__(self, stream, directives, ctxt, **vars):
        ctxt._match_templates.append((self.path.test(ignore_context=True), self.path, list(stream), self.hints, self.namespaces, directives))
        return []

    def __repr__(self):
        return '<%s "%s">' % (type(self).__name__, self.path.source)