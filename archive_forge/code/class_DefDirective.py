import six
from genshi.core import QName, Stream
from genshi.path import Path
from genshi.template.base import TemplateRuntimeError, TemplateSyntaxError, \
from genshi.template.eval import Expression, _ast, _parse
class DefDirective(Directive):
    """Implementation of the ``py:def`` template directive.
    
    This directive can be used to create "Named Template Functions", which
    are template snippets that are not actually output during normal
    processing, but rather can be expanded from expressions in other places
    in the template.
    
    A named template function can be used just like a normal Python function
    from template expressions:
    
    >>> from genshi.template import MarkupTemplate
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <p py:def="echo(greeting, name='world')" class="message">
    ...     ${greeting}, ${name}!
    ...   </p>
    ...   ${echo('Hi', name='you')}
    ... </div>''')
    >>> print(tmpl.generate(bar='Bye'))
    <div>
      <p class="message">
        Hi, you!
      </p>
    </div>
    
    If a function does not require parameters, the parenthesis can be omitted
    in the definition:
    
    >>> tmpl = MarkupTemplate('''<div xmlns:py="http://genshi.edgewall.org/">
    ...   <p py:def="helloworld" class="message">
    ...     Hello, world!
    ...   </p>
    ...   ${helloworld()}
    ... </div>''')
    >>> print(tmpl.generate(bar='Bye'))
    <div>
      <p class="message">
        Hello, world!
      </p>
    </div>
    """
    __slots__ = ['name', 'args', 'star_args', 'dstar_args', 'defaults']

    def __init__(self, args, template, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        ast = _parse(args).body
        self.args = []
        self.star_args = None
        self.dstar_args = None
        self.defaults = {}
        if isinstance(ast, _ast.Call):
            self.name = ast.func.id
            for arg in ast.args:
                if hasattr(_ast, 'Starred') and isinstance(arg, _ast.Starred):
                    self.star_args = arg.value.id
                else:
                    self.args.append(arg.id)
            for kwd in ast.keywords:
                if kwd.arg is None:
                    self.dstar_args = kwd.value.id
                else:
                    self.args.append(kwd.arg)
                    exp = Expression(kwd.value, template.filepath, lineno, lookup=template.lookup)
                    self.defaults[kwd.arg] = exp
            if getattr(ast, 'starargs', None):
                self.star_args = ast.starargs.id
            if getattr(ast, 'kwargs', None):
                self.dstar_args = ast.kwargs.id
        else:
            self.name = ast.id

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            value = value.get('function')
        return super(DefDirective, cls).attach(template, stream, value, namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        stream = list(stream)

        def function(*args, **kwargs):
            scope = {}
            args = list(args)
            for name in self.args:
                if args:
                    scope[name] = args.pop(0)
                else:
                    if name in kwargs:
                        val = kwargs.pop(name)
                    else:
                        val = _eval_expr(self.defaults.get(name), ctxt, vars)
                    scope[name] = val
            if not self.star_args is None:
                scope[self.star_args] = args
            if not self.dstar_args is None:
                scope[self.dstar_args] = kwargs
            ctxt.push(scope)
            for event in _apply_directives(stream, directives, ctxt, vars):
                yield event
            ctxt.pop()
        function.__name__ = self.name
        ctxt.frames[-1][self.name] = function
        return []

    def __repr__(self):
        return '<%s "%s">' % (type(self).__name__, self.name)