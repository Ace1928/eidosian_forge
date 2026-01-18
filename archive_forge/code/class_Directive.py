import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
@six.add_metaclass(DirectiveMeta)
class Directive(object):
    """Abstract base class for template directives.
    
    A directive is basically a callable that takes three positional arguments:
    ``ctxt`` is the template data context, ``stream`` is an iterable over the
    events that the directive applies to, and ``directives`` is is a list of
    other directives on the same stream that need to be applied.
    
    Directives can be "anonymous" or "registered". Registered directives can be
    applied by the template author using an XML attribute with the
    corresponding name in the template. Such directives should be subclasses of
    this base class that can  be instantiated with the value of the directive
    attribute as parameter.
    
    Anonymous directives are simply functions conforming to the protocol
    described above, and can only be applied programmatically (for example by
    template filters).
    """
    __slots__ = ['expr']

    def __init__(self, value, template=None, namespaces=None, lineno=-1, offset=-1):
        self.expr = self._parse_expr(value, template, lineno, offset)

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        """Called after the template stream has been completely parsed.
        
        :param template: the `Template` object
        :param stream: the event stream associated with the directive
        :param value: the argument value for the directive; if the directive was
                      specified as an element, this will be an `Attrs` instance
                      with all specified attributes, otherwise it will be a
                      `unicode` object with just the attribute value
        :param namespaces: a mapping of namespace URIs to prefixes
        :param pos: a ``(filename, lineno, offset)`` tuple describing the
                    location where the directive was found in the source
        
        This class method should return a ``(directive, stream)`` tuple. If
        ``directive`` is not ``None``, it should be an instance of the `Directive`
        class, and gets added to the list of directives applied to the substream
        at runtime. `stream` is an event stream that replaces the original
        stream associated with the directive.
        """
        return (cls(value, template, namespaces, *pos[1:]), stream)

    def __call__(self, stream, directives, ctxt, **vars):
        """Apply the directive to the given stream.
        
        :param stream: the event stream
        :param directives: a list of the remaining directives that should
                           process the stream
        :param ctxt: the context data
        :param vars: additional variables that should be made available when
                     Python code is executed
        """
        raise NotImplementedError

    def __repr__(self):
        expr = ''
        if getattr(self, 'expr', None) is not None:
            expr = ' "%s"' % self.expr.source
        return '<%s%s>' % (type(self).__name__, expr)

    @classmethod
    def _parse_expr(cls, expr, template, lineno=-1, offset=-1):
        """Parses the given expression, raising a useful error message when a
        syntax error is encountered.
        """
        try:
            return expr and Expression(expr, template.filepath, lineno, lookup=template.lookup) or None
        except SyntaxError as err:
            err.msg += ' in expression "%s" of "%s" directive' % (expr, cls.tagname)
            raise TemplateSyntaxError(err, template.filepath, lineno, offset + (err.offset or 0))