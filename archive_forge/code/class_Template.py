from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
class Template(DirectiveFactory):
    """Abstract template base class.
    
    This class implements most of the template processing model, but does not
    specify the syntax of templates.
    """
    EXEC = StreamEventKind('EXEC')
    'Stream event kind representing a Python code suite to execute.'
    EXPR = StreamEventKind('EXPR')
    'Stream event kind representing a Python expression.'
    INCLUDE = StreamEventKind('INCLUDE')
    'Stream event kind representing the inclusion of another template.'
    SUB = StreamEventKind('SUB')
    'Stream event kind representing a nested stream to which one or more\n    directives should be applied.\n    '
    serializer = None
    _number_conv = six.text_type

    def __init__(self, source, filepath=None, filename=None, loader=None, encoding=None, lookup='strict', allow_exec=True):
        """Initialize a template from either a string, a file-like object, or
        an already parsed markup stream.
        
        :param source: a string, file-like object, or markup stream to read the
                       template from
        :param filepath: the absolute path to the template file
        :param filename: the path to the template file relative to the search
                         path
        :param loader: the `TemplateLoader` to use for loading included
                       templates
        :param encoding: the encoding of the `source`
        :param lookup: the variable lookup mechanism; either "strict" (the
                       default), "lenient", or a custom lookup class
        :param allow_exec: whether Python code blocks in templates should be
                           allowed
        
        :note: Changed in 0.5: Added the `allow_exec` argument
        """
        self.filepath = filepath or filename
        self.filename = filename
        self.loader = loader
        self.lookup = lookup
        self.allow_exec = allow_exec
        self._init_filters()
        self._init_loader()
        self._prepared = False
        if not isinstance(source, Stream) and (not hasattr(source, 'read')):
            if isinstance(source, six.text_type):
                source = StringIO(source)
            else:
                source = BytesIO(source)
        try:
            self._stream = self._parse(source, encoding)
        except ParseError as e:
            raise TemplateSyntaxError(e.msg, self.filepath, e.lineno, e.offset)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['filters'] = []
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._init_filters()

    def __repr__(self):
        return '<%s "%s">' % (type(self).__name__, self.filename)

    def _init_filters(self):
        self.filters = [self._flatten, self._include]

    def _init_loader(self):
        if self.loader is None:
            from genshi.template.loader import TemplateLoader
            if self.filename:
                if self.filepath != self.filename:
                    basedir = os.path.normpath(self.filepath)[:-len(os.path.normpath(self.filename))]
                else:
                    basedir = os.path.dirname(self.filename)
            else:
                basedir = '.'
            self.loader = TemplateLoader([os.path.abspath(basedir)])

    @property
    def stream(self):
        if not self._prepared:
            self._prepare_self()
        return self._stream

    def _parse(self, source, encoding):
        """Parse the template.
        
        The parsing stage parses the template and constructs a list of
        directives that will be executed in the render stage. The input is
        split up into literal output (text that does not depend on the context
        data) and directives or expressions.
        
        :param source: a file-like object containing the XML source of the
                       template, or an XML event stream
        :param encoding: the encoding of the `source`
        """
        raise NotImplementedError

    def _prepare_self(self, inlined=None):
        if not self._prepared:
            self._stream = list(self._prepare(self._stream, inlined))
            self._prepared = True

    def _prepare(self, stream, inlined):
        """Call the `attach` method of every directive found in the template.
        
        :param stream: the event stream of the template
        """
        from genshi.template.loader import TemplateNotFound
        if inlined is None:
            inlined = set((self.filepath,))
        for kind, data, pos in stream:
            if kind is SUB:
                directives = []
                substream = data[1]
                for _, cls, value, namespaces, pos in sorted(data[0], key=lambda x: x[0]):
                    directive, substream = cls.attach(self, substream, value, namespaces, pos)
                    if directive:
                        directives.append(directive)
                substream = self._prepare(substream, inlined)
                if directives:
                    yield (kind, (directives, list(substream)), pos)
                else:
                    for event in substream:
                        yield event
            elif kind is INCLUDE:
                href, cls, fallback = data
                tmpl_inlined = False
                if isinstance(href, six.string_types) and (not getattr(self.loader, 'auto_reload', True)):
                    tmpl = None
                    try:
                        tmpl = self.loader.load(href, relative_to=pos[0], cls=cls or self.__class__)
                    except TemplateNotFound:
                        if fallback is None:
                            raise
                    if tmpl is not None:
                        if tmpl.filepath not in inlined:
                            inlined.add(tmpl.filepath)
                            tmpl._prepare_self(inlined)
                            for event in tmpl.stream:
                                yield event
                            inlined.discard(tmpl.filepath)
                            tmpl_inlined = True
                    else:
                        for event in self._prepare(fallback, inlined):
                            yield event
                        tmpl_inlined = True
                if tmpl_inlined:
                    continue
                if fallback:
                    data = (href, cls, list(self._prepare(fallback, inlined)))
                yield (kind, data, pos)
            else:
                yield (kind, data, pos)

    def generate(self, *args, **kwargs):
        """Apply the template to the given context data.
        
        Any keyword arguments are made available to the template as context
        data.
        
        Only one positional argument is accepted: if it is provided, it must be
        an instance of the `Context` class, and keyword arguments are ignored.
        This calling style is used for internal processing.
        
        :return: a markup event stream representing the result of applying
                 the template to the context data.
        """
        vars = {}
        if args:
            assert len(args) == 1
            ctxt = args[0]
            if ctxt is None:
                ctxt = Context(**kwargs)
            else:
                vars = kwargs
            assert isinstance(ctxt, Context)
        else:
            ctxt = Context(**kwargs)
        stream = self.stream
        for filter_ in self.filters:
            stream = filter_(iter(stream), ctxt, **vars)
        return Stream(stream, self.serializer)

    def _flatten(self, stream, ctxt, **vars):
        number_conv = self._number_conv
        stack = []
        push = stack.append
        pop = stack.pop
        stream = iter(stream)
        while 1:
            for kind, data, pos in stream:
                if kind is START and data[1]:
                    tag, attrs = data
                    new_attrs = []
                    for name, value in attrs:
                        if type(value) is list:
                            values = [event[1] for event in self._flatten(value, ctxt, **vars) if event[0] is TEXT and event[1] is not None]
                            if not values:
                                continue
                            value = ''.join(values)
                        new_attrs.append((name, value))
                    yield (kind, (tag, Attrs(new_attrs)), pos)
                elif kind is EXPR:
                    result = _eval_expr(data, ctxt, vars)
                    if result is not None:
                        if isinstance(result, six.string_types):
                            yield (TEXT, result, pos)
                        elif isinstance(result, numeric_types):
                            yield (TEXT, number_conv(result), pos)
                        elif hasattr(result, '__iter__'):
                            push(stream)
                            stream = _ensure(result)
                            break
                        else:
                            yield (TEXT, six.text_type(result), pos)
                elif kind is SUB:
                    push(stream)
                    stream = _apply_directives(data[1], data[0], ctxt, vars)
                    break
                elif kind is EXEC:
                    _exec_suite(data, ctxt, vars)
                else:
                    yield (kind, data, pos)
            else:
                if not stack:
                    break
                stream = pop()

    def _include(self, stream, ctxt, **vars):
        """Internal stream filter that performs inclusion of external
        template files.
        """
        from genshi.template.loader import TemplateNotFound
        for event in stream:
            if event[0] is INCLUDE:
                href, cls, fallback = event[1]
                if not isinstance(href, six.string_types):
                    parts = []
                    for subkind, subdata, subpos in self._flatten(href, ctxt, **vars):
                        if subkind is TEXT:
                            parts.append(subdata)
                    href = ''.join([x for x in parts if x is not None])
                try:
                    tmpl = self.loader.load(href, relative_to=event[2][0], cls=cls or self.__class__)
                    for event in tmpl.generate(ctxt, **vars):
                        yield event
                except TemplateNotFound:
                    if fallback is None:
                        raise
                    for filter_ in self.filters:
                        fallback = filter_(iter(fallback), ctxt, **vars)
                    for event in fallback:
                        yield event
            else:
                yield event