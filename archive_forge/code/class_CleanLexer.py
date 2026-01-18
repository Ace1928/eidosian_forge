from pygments.lexer import ExtendedRegexLexer, LexerContext, \
from pygments.token import Comment, Keyword, Literal, Name, Number, Operator, \
class CleanLexer(ExtendedRegexLexer):
    """
    Lexer for the general purpose, state-of-the-art, pure and lazy functional
    programming language Clean (http://clean.cs.ru.nl/Clean).

    .. versionadded: 2.2
    """
    name = 'Clean'
    aliases = ['clean']
    filenames = ['*.icl', '*.dcl']

    def get_tokens_unprocessed(self, text=None, context=None):
        ctx = LexerContext(text, 0)
        ctx.indent = 0
        return ExtendedRegexLexer.get_tokens_unprocessed(self, text, context=ctx)

    def check_class_not_import(lexer, match, ctx):
        if match.group(0) == 'import':
            yield (match.start(), Keyword.Namespace, match.group(0))
            ctx.stack = ctx.stack[:-1] + ['fromimportfunc']
        else:
            yield (match.start(), Name.Class, match.group(0))
        ctx.pos = match.end()

    def check_instance_class(lexer, match, ctx):
        if match.group(0) == 'instance' or match.group(0) == 'class':
            yield (match.start(), Keyword, match.group(0))
        else:
            yield (match.start(), Name.Function, match.group(0))
            ctx.stack = ctx.stack + ['fromimportfunctype']
        ctx.pos = match.end()

    @staticmethod
    def indent_len(text):
        text = text.replace('\n', '')
        return (len(text.replace('\t', '    ')), len(text))

    def store_indent(lexer, match, ctx):
        ctx.indent, _ = CleanLexer.indent_len(match.group(0))
        ctx.pos = match.end()
        yield (match.start(), Text, match.group(0))

    def check_indent1(lexer, match, ctx):
        indent, reallen = CleanLexer.indent_len(match.group(0))
        if indent > ctx.indent:
            yield (match.start(), Whitespace, match.group(0))
            ctx.pos = match.start() + reallen + 1
        else:
            ctx.indent = 0
            ctx.pos = match.start()
            ctx.stack = ctx.stack[:-1]
            yield (match.start(), Whitespace, match.group(0)[1:])

    def check_indent2(lexer, match, ctx):
        indent, reallen = CleanLexer.indent_len(match.group(0))
        if indent > ctx.indent:
            yield (match.start(), Whitespace, match.group(0))
            ctx.pos = match.start() + reallen + 1
        else:
            ctx.indent = 0
            ctx.pos = match.start()
            ctx.stack = ctx.stack[:-2]

    def check_indent3(lexer, match, ctx):
        indent, reallen = CleanLexer.indent_len(match.group(0))
        if indent > ctx.indent:
            yield (match.start(), Whitespace, match.group(0))
            ctx.pos = match.start() + reallen + 1
        else:
            ctx.indent = 0
            ctx.pos = match.start()
            ctx.stack = ctx.stack[:-3]
            yield (match.start(), Whitespace, match.group(0)[1:])
            if match.group(0) == '\n\n':
                ctx.pos = ctx.pos + 1

    def skip(lexer, match, ctx):
        ctx.stack = ctx.stack[:-1]
        ctx.pos = match.end()
        yield (match.start(), Comment, match.group(0))
    keywords = ('class', 'instance', 'where', 'with', 'let', 'let!', 'in', 'case', 'of', 'infix', 'infixr', 'infixl', 'generic', 'derive', 'otherwise', 'code', 'inline')
    tokens = {'common': [(';', Punctuation, '#pop'), ('//', Comment, 'singlecomment')], 'root': [('//.*\\n', Comment.Single), ('(?s)/\\*\\*.*?\\*/', Comment.Special), ('(?s)/\\*.*?\\*/', Comment.Multi), ('\\b((?:implementation|definition|system)\\s+)?(module)(\\s+)([\\w`.]+)', bygroups(Keyword.Namespace, Keyword.Namespace, Text, Name.Class)), ('(?<=\\n)import(?=\\s)', Keyword.Namespace, 'import'), ('(?<=\\n)from(?=\\s)', Keyword.Namespace, 'fromimport'), (words(keywords, prefix='(?<=\\s)', suffix='(?=\\s)'), Keyword), (words(keywords, prefix='^', suffix='(?=\\s)'), Keyword), ('(?=\\{\\|)', Whitespace, 'genericfunction'), ('(?<=\\n)([ \\t]*)([\\w`$()=\\-<>~*\\^|+&%]+)((?:\\s+\\w)*)(\\s*)(::)', bygroups(store_indent, Name.Function, Keyword.Type, Whitespace, Punctuation), 'functiondefargs'), ('(?<=\\n)([ \\t]*)(::)', bygroups(store_indent, Punctuation), 'typedef'), ('^([ \\t]*)(::)', bygroups(store_indent, Punctuation), 'typedef'), ("\\'\\\\?.(?<!\\\\)\\'", String.Char), ("\\'\\\\\\d+\\'", String.Char), ("\\'\\\\\\\\\\'", String.Char), ('[+\\-~]?\\s*\\d+\\.\\d+(E[+\\-~]?\\d+)?\\b', Number.Float), ('[+\\-~]?\\s*0[0-7]\\b', Number.Oct), ('[+\\-~]?\\s*0x[0-9a-fA-F]\\b', Number.Hex), ('[+\\-~]?\\s*\\d+\\b', Number.Integer), ('"', String.Double, 'doubleqstring'), (words(('True', 'False'), prefix='(?<=\\s)', suffix='(?=\\s)'), Literal), ("(\\')([\\w.]+)(\\'\\.)", bygroups(Punctuation, Name.Namespace, Punctuation)), ('([\\w`$%/?@]+\\.?)*[\\w`$%/?@]+', Name), ('[{}()\\[\\],:;.#]', Punctuation), ('[+\\-=!<>|&~*\\^/]', Operator), ('\\\\\\\\', Operator), ('\\\\.*?(->|\\.|=)', Name.Function), ('\\s', Whitespace), include('common')], 'fromimport': [include('common'), ('([\\w`.]+)', check_class_not_import), ('\\n', Whitespace, '#pop'), ('\\s', Whitespace)], 'fromimportfunc': [include('common'), ('(::)(\\s+)([^,\\s]+)', bygroups(Punctuation, Text, Keyword.Type)), ('([\\w`$()=\\-<>~*\\^|+&%/]+)', check_instance_class), (',', Punctuation), ('\\n', Whitespace, '#pop'), ('\\s', Whitespace)], 'fromimportfunctype': [include('common'), ('[{(\\[]', Punctuation, 'combtype'), (',', Punctuation, '#pop'), ('[:;.#]', Punctuation), ('\\n', Whitespace, '#pop:2'), ('[^\\S\\n]+', Whitespace), ('\\S+', Keyword.Type)], 'combtype': [include('common'), ('[})\\]]', Punctuation, '#pop'), ('[{(\\[]', Punctuation, '#pop'), ('[,:;.#]', Punctuation), ('\\s+', Whitespace), ('\\S+', Keyword.Type)], 'import': [include('common'), (words(('from', 'import', 'as', 'qualified'), prefix='(?<=\\s)', suffix='(?=\\s)'), Keyword.Namespace), ('[\\w`.]+', Name.Class), ('\\n', Whitespace, '#pop'), (',', Punctuation), ('[^\\S\\n]+', Whitespace)], 'singlecomment': [('(.)(?=\\n)', skip), ('.+(?!\\n)', Comment)], 'doubleqstring': [('[^\\\\"]+', String.Double), ('"', String.Double, '#pop'), ('\\\\.', String.Double)], 'typedef': [include('common'), ('[\\w`]+', Keyword.Type), ('[:=|(),\\[\\]{}!*]', Punctuation), ('->', Punctuation), ('\\n(?=[^\\s|])', Whitespace, '#pop'), ('\\s', Whitespace), ('.', Keyword.Type)], 'genericfunction': [include('common'), ('\\{\\|', Punctuation), ('\\|\\}', Punctuation, '#pop'), (',', Punctuation), ('->', Punctuation), ('(\\s+of\\s+)(\\{)', bygroups(Keyword, Punctuation), 'genericftypes'), ('\\s', Whitespace), ('[\\w`\\[\\]{}!]+', Keyword.Type), ('[*()]', Punctuation)], 'genericftypes': [include('common'), ('[\\w`]+', Keyword.Type), (',', Punctuation), ('\\s', Whitespace), ('\\}', Punctuation, '#pop')], 'functiondefargs': [include('common'), ('\\n(\\s*)', check_indent1), ('[!{}()\\[\\],:;.#]', Punctuation), ('->', Punctuation, 'functiondefres'), ('^(?=\\S)', Whitespace, '#pop'), ('\\S', Keyword.Type), ('\\s', Whitespace)], 'functiondefres': [include('common'), ('\\n(\\s*)', check_indent2), ('^(?=\\S)', Whitespace, '#pop:2'), ('[!{}()\\[\\],:;.#]', Punctuation), ('\\|', Punctuation, 'functiondefclasses'), ('\\S', Keyword.Type), ('\\s', Whitespace)], 'functiondefclasses': [include('common'), ('\\n(\\s*)', check_indent3), ('^(?=\\S)', Whitespace, '#pop:3'), ('[,&]', Punctuation), ('\\[', Punctuation, 'functiondefuniquneq'), ('[\\w`$()=\\-<>~*\\^|+&%/{}\\[\\]@]', Name.Function, 'functionname'), ('\\s+', Whitespace)], 'functiondefuniquneq': [include('common'), ('[a-z]+', Keyword.Type), ('\\s+', Whitespace), ('<=|,', Punctuation), ('\\]', Punctuation, '#pop')], 'functionname': [include('common'), ('[\\w`$()=\\-<>~*\\^|+&%/]+', Name.Function), ('(?=\\{\\|)', Punctuation, 'genericfunction'), default('#pop')]}