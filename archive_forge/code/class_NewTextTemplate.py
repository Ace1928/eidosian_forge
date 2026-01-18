import re
import six
from genshi.core import TEXT
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.directives import *
from genshi.template.interpolation import interpolate
class NewTextTemplate(Template):
    """Implementation of a simple text-based template engine. This class will
    replace `OldTextTemplate` in a future release.
    
    It uses a more explicit delimiting style for directives: instead of the old
    style which required putting directives on separate lines that were prefixed
    with a ``#`` sign, directives and commenbtsr are enclosed in delimiter pairs
    (by default ``{% ... %}`` and ``{# ... #}``, respectively).
    
    Variable substitution uses the same interpolation syntax as for markup
    languages: simple references are prefixed with a dollar sign, more complex
    expression enclosed in curly braces.
    
    >>> tmpl = NewTextTemplate('''Dear $name,
    ... 
    ... {# This is a comment #}
    ... We have the following items for you:
    ... {% for item in items %}
    ...  * ${'Item %d' % item}
    ... {% end %}
    ... ''')
    >>> print(tmpl.generate(name='Joe', items=[1, 2, 3]).render(encoding=None))
    Dear Joe,
    <BLANKLINE>
    <BLANKLINE>
    We have the following items for you:
    <BLANKLINE>
     * Item 1
    <BLANKLINE>
     * Item 2
    <BLANKLINE>
     * Item 3
    <BLANKLINE>
    <BLANKLINE>
    
    By default, no spaces or line breaks are removed. If a line break should
    not be included in the output, prefix it with a backslash:
    
    >>> tmpl = NewTextTemplate('''Dear $name,
    ... 
    ... {# This is a comment #}\\
    ... We have the following items for you:
    ... {% for item in items %}\\
    ...  * $item
    ... {% end %}\\
    ... ''')
    >>> print(tmpl.generate(name='Joe', items=[1, 2, 3]).render(encoding=None))
    Dear Joe,
    <BLANKLINE>
    We have the following items for you:
     * 1
     * 2
     * 3
    <BLANKLINE>
    
    Backslashes are also used to escape the start delimiter of directives and
    comments:

    >>> tmpl = NewTextTemplate('''Dear $name,
    ... 
    ... \\\\{# This is a comment #}
    ... We have the following items for you:
    ... {% for item in items %}\\
    ...  * $item
    ... {% end %}\\
    ... ''')
    >>> print(tmpl.generate(name='Joe', items=[1, 2, 3]).render(encoding=None))
    Dear Joe,
    <BLANKLINE>
    {# This is a comment #}
    We have the following items for you:
     * 1
     * 2
     * 3
    <BLANKLINE>
    
    :since: version 0.5
    """
    directives = [('def', DefDirective), ('when', WhenDirective), ('otherwise', OtherwiseDirective), ('for', ForDirective), ('if', IfDirective), ('choose', ChooseDirective), ('with', WithDirective)]
    serializer = 'text'
    _DIRECTIVE_RE = '((?<!\\\\)%s\\s*(\\w+)\\s*(.*?)\\s*%s|(?<!\\\\)%s.*?%s)'
    _ESCAPE_RE = '\\\\\\n|\\\\\\r\\n|\\\\(\\\\)|\\\\(%s)|\\\\(%s)'

    def __init__(self, source, filepath=None, filename=None, loader=None, encoding=None, lookup='strict', allow_exec=False, delims=('{%', '%}', '{#', '#}')):
        self.delimiters = delims
        Template.__init__(self, source, filepath=filepath, filename=filename, loader=loader, encoding=encoding, lookup=lookup)

    def _get_delims(self):
        return self._delims

    def _set_delims(self, delims):
        if len(delims) != 4:
            raise ValueError('delimiers tuple must have exactly four elements')
        self._delims = delims
        self._directive_re = re.compile(self._DIRECTIVE_RE % tuple([re.escape(d) for d in delims]), re.DOTALL)
        self._escape_re = re.compile(self._ESCAPE_RE % tuple([re.escape(d) for d in delims[::2]]))
    delimiters = property(_get_delims, _set_delims, '    The delimiters for directives and comments. This should be a four item tuple\n    of the form ``(directive_start, directive_end, comment_start,\n    comment_end)``, where each item is a string.\n    ')

    def _parse(self, source, encoding):
        """Parse the template from text input."""
        stream = []
        dirmap = {}
        depth = 0
        source = source.read()
        if not isinstance(source, six.text_type):
            source = source.decode(encoding or 'utf-8', 'replace')
        offset = 0
        lineno = 1
        _escape_sub = self._escape_re.sub

        def _escape_repl(mo):
            groups = [g for g in mo.groups() if g]
            if not groups:
                return ''
            return groups[0]
        for idx, mo in enumerate(self._directive_re.finditer(source)):
            start, end = mo.span(1)
            if start > offset:
                text = _escape_sub(_escape_repl, source[offset:start])
                for kind, data, pos in interpolate(text, self.filepath, lineno, lookup=self.lookup):
                    stream.append((kind, data, pos))
                lineno += len(text.splitlines())
            lineno += len(source[start:end].splitlines())
            command, value = mo.group(2, 3)
            if command == 'include':
                pos = (self.filename, lineno, 0)
                value = list(interpolate(value, self.filepath, lineno, 0, lookup=self.lookup))
                if len(value) == 1 and value[0][0] is TEXT:
                    value = value[0][1]
                stream.append((INCLUDE, (value, None, []), pos))
            elif command == 'python':
                if not self.allow_exec:
                    raise TemplateSyntaxError('Python code blocks not allowed', self.filepath, lineno)
                try:
                    suite = Suite(value, self.filepath, lineno, lookup=self.lookup)
                except SyntaxError as err:
                    raise TemplateSyntaxError(err, self.filepath, lineno + (err.lineno or 1) - 1)
                pos = (self.filename, lineno, 0)
                stream.append((EXEC, suite, pos))
            elif command == 'end':
                depth -= 1
                if depth in dirmap:
                    directive, start_offset = dirmap.pop(depth)
                    substream = stream[start_offset:]
                    stream[start_offset:] = [(SUB, ([directive], substream), (self.filepath, lineno, 0))]
            elif command:
                cls = self.get_directive(command)
                if cls is None:
                    raise BadDirectiveError(command)
                directive = (0, cls, value, None, (self.filepath, lineno, 0))
                dirmap[depth] = (directive, len(stream))
                depth += 1
            offset = end
        if offset < len(source):
            text = _escape_sub(_escape_repl, source[offset:])
            for kind, data, pos in interpolate(text, self.filepath, lineno, lookup=self.lookup):
                stream.append((kind, data, pos))
        return stream