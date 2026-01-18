import re
from pygments.lexer import RegexLexer, include, bygroups, default, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class SMLLexer(RegexLexer):
    """
    For the Standard ML language.

    .. versionadded:: 1.5
    """
    name = 'Standard ML'
    aliases = ['sml']
    filenames = ['*.sml', '*.sig', '*.fun']
    mimetypes = ['text/x-standardml', 'application/x-standardml']
    alphanumid_reserved = set(('abstype', 'and', 'andalso', 'as', 'case', 'datatype', 'do', 'else', 'end', 'exception', 'fn', 'fun', 'handle', 'if', 'in', 'infix', 'infixr', 'let', 'local', 'nonfix', 'of', 'op', 'open', 'orelse', 'raise', 'rec', 'then', 'type', 'val', 'with', 'withtype', 'while', 'eqtype', 'functor', 'include', 'sharing', 'sig', 'signature', 'struct', 'structure', 'where'))
    symbolicid_reserved = set((':', '\\|', '=', '=>', '->', '#', ':>'))
    nonid_reserved = set(('(', ')', '[', ']', '{', '}', ',', ';', '...', '_'))
    alphanumid_re = "[a-zA-Z][\\w']*"
    symbolicid_re = '[!%&$#+\\-/:<=>?@\\\\~`^|*]+'

    def stringy(whatkind):
        return [('[^"\\\\]', whatkind), ('\\\\[\\\\"abtnvfr]', String.Escape), ('\\\\\\^[\\x40-\\x5e]', String.Escape), ('\\\\[0-9]{3}', String.Escape), ('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\\\s+\\\\', String.Interpol), ('"', whatkind, '#pop')]

    def long_id_callback(self, match):
        if match.group(1) in self.alphanumid_reserved:
            token = Error
        else:
            token = Name.Namespace
        yield (match.start(1), token, match.group(1))
        yield (match.start(2), Punctuation, match.group(2))

    def end_id_callback(self, match):
        if match.group(1) in self.alphanumid_reserved:
            token = Error
        elif match.group(1) in self.symbolicid_reserved:
            token = Error
        else:
            token = Name
        yield (match.start(1), token, match.group(1))

    def id_callback(self, match):
        str = match.group(1)
        if str in self.alphanumid_reserved:
            token = Keyword.Reserved
        elif str in self.symbolicid_reserved:
            token = Punctuation
        else:
            token = Name
        yield (match.start(1), token, str)
    tokens = {'whitespace': [('\\s+', Text), ('\\(\\*', Comment.Multiline, 'comment')], 'delimiters': [('\\(|\\[|\\{', Punctuation, 'main'), ('\\)|\\]|\\}', Punctuation, '#pop'), ("\\b(let|if|local)\\b(?!\\')", Keyword.Reserved, ('main', 'main')), ("\\b(struct|sig|while)\\b(?!\\')", Keyword.Reserved, 'main'), ("\\b(do|else|end|in|then)\\b(?!\\')", Keyword.Reserved, '#pop')], 'core': [('(%s)' % '|'.join((re.escape(z) for z in nonid_reserved)), Punctuation), ('#"', String.Char, 'char'), ('"', String.Double, 'string'), ('~?0x[0-9a-fA-F]+', Number.Hex), ('0wx[0-9a-fA-F]+', Number.Hex), ('0w\\d+', Number.Integer), ('~?\\d+\\.\\d+[eE]~?\\d+', Number.Float), ('~?\\d+\\.\\d+', Number.Float), ('~?\\d+[eE]~?\\d+', Number.Float), ('~?\\d+', Number.Integer), ('#\\s*[1-9][0-9]*', Name.Label), ('#\\s*(%s)' % alphanumid_re, Name.Label), ('#\\s+(%s)' % symbolicid_re, Name.Label), ("\\b(datatype|abstype)\\b(?!\\')", Keyword.Reserved, 'dname'), ("(?=\\b(exception)\\b(?!\\'))", Text, 'ename'), ("\\b(functor|include|open|signature|structure)\\b(?!\\')", Keyword.Reserved, 'sname'), ("\\b(type|eqtype)\\b(?!\\')", Keyword.Reserved, 'tname'), ("\\'[\\w\\']*", Name.Decorator), ('(%s)(\\.)' % alphanumid_re, long_id_callback, 'dotted'), ('(%s)' % alphanumid_re, id_callback), ('(%s)' % symbolicid_re, id_callback)], 'dotted': [('(%s)(\\.)' % alphanumid_re, long_id_callback), ('(%s)' % alphanumid_re, end_id_callback, '#pop'), ('(%s)' % symbolicid_re, end_id_callback, '#pop'), ('\\s+', Error), ('\\S+', Error)], 'root': [default('main')], 'main': [include('whitespace'), ("\\b(val|and)\\b(?!\\')", Keyword.Reserved, 'vname'), ("\\b(fun)\\b(?!\\')", Keyword.Reserved, ('#pop', 'main-fun', 'fname')), include('delimiters'), include('core'), ('\\S+', Error)], 'main-fun': [include('whitespace'), ('\\s', Text), ('\\(\\*', Comment.Multiline, 'comment'), ("\\b(fun|and)\\b(?!\\')", Keyword.Reserved, 'fname'), ("\\b(val)\\b(?!\\')", Keyword.Reserved, ('#pop', 'main', 'vname')), ('\\|', Punctuation, 'fname'), ("\\b(case|handle)\\b(?!\\')", Keyword.Reserved, ('#pop', 'main')), include('delimiters'), include('core'), ('\\S+', Error)], 'char': stringy(String.Char), 'string': stringy(String.Double), 'breakout': [("(?=\\b(%s)\\b(?!\\'))" % '|'.join(alphanumid_reserved), Text, '#pop')], 'sname': [include('whitespace'), include('breakout'), ('(%s)' % alphanumid_re, Name.Namespace), default('#pop')], 'fname': [include('whitespace'), ("\\'[\\w\\']*", Name.Decorator), ('\\(', Punctuation, 'tyvarseq'), ('(%s)' % alphanumid_re, Name.Function, '#pop'), ('(%s)' % symbolicid_re, Name.Function, '#pop'), default('#pop')], 'vname': [include('whitespace'), ("\\'[\\w\\']*", Name.Decorator), ('\\(', Punctuation, 'tyvarseq'), ('(%s)(\\s*)(=(?!%s))' % (alphanumid_re, symbolicid_re), bygroups(Name.Variable, Text, Punctuation), '#pop'), ('(%s)(\\s*)(=(?!%s))' % (symbolicid_re, symbolicid_re), bygroups(Name.Variable, Text, Punctuation), '#pop'), ('(%s)' % alphanumid_re, Name.Variable, '#pop'), ('(%s)' % symbolicid_re, Name.Variable, '#pop'), default('#pop')], 'tname': [include('whitespace'), include('breakout'), ("\\'[\\w\\']*", Name.Decorator), ('\\(', Punctuation, 'tyvarseq'), ('=(?!%s)' % symbolicid_re, Punctuation, ('#pop', 'typbind')), ('(%s)' % alphanumid_re, Keyword.Type), ('(%s)' % symbolicid_re, Keyword.Type), ('\\S+', Error, '#pop')], 'typbind': [include('whitespace'), ("\\b(and)\\b(?!\\')", Keyword.Reserved, ('#pop', 'tname')), include('breakout'), include('core'), ('\\S+', Error, '#pop')], 'dname': [include('whitespace'), include('breakout'), ("\\'[\\w\\']*", Name.Decorator), ('\\(', Punctuation, 'tyvarseq'), ('(=)(\\s*)(datatype)', bygroups(Punctuation, Text, Keyword.Reserved), '#pop'), ('=(?!%s)' % symbolicid_re, Punctuation, ('#pop', 'datbind', 'datcon')), ('(%s)' % alphanumid_re, Keyword.Type), ('(%s)' % symbolicid_re, Keyword.Type), ('\\S+', Error, '#pop')], 'datbind': [include('whitespace'), ("\\b(and)\\b(?!\\')", Keyword.Reserved, ('#pop', 'dname')), ("\\b(withtype)\\b(?!\\')", Keyword.Reserved, ('#pop', 'tname')), ("\\b(of)\\b(?!\\')", Keyword.Reserved), ('(\\|)(\\s*)(%s)' % alphanumid_re, bygroups(Punctuation, Text, Name.Class)), ('(\\|)(\\s+)(%s)' % symbolicid_re, bygroups(Punctuation, Text, Name.Class)), include('breakout'), include('core'), ('\\S+', Error)], 'ename': [include('whitespace'), ('(exception|and)\\b(\\s+)(%s)' % alphanumid_re, bygroups(Keyword.Reserved, Text, Name.Class)), ('(exception|and)\\b(\\s*)(%s)' % symbolicid_re, bygroups(Keyword.Reserved, Text, Name.Class)), ("\\b(of)\\b(?!\\')", Keyword.Reserved), include('breakout'), include('core'), ('\\S+', Error)], 'datcon': [include('whitespace'), ('(%s)' % alphanumid_re, Name.Class, '#pop'), ('(%s)' % symbolicid_re, Name.Class, '#pop'), ('\\S+', Error, '#pop')], 'tyvarseq': [('\\s', Text), ('\\(\\*', Comment.Multiline, 'comment'), ("\\'[\\w\\']*", Name.Decorator), (alphanumid_re, Name), (',', Punctuation), ('\\)', Punctuation, '#pop'), (symbolicid_re, Name)], 'comment': [('[^(*)]', Comment.Multiline), ('\\(\\*', Comment.Multiline, '#push'), ('\\*\\)', Comment.Multiline, '#pop'), ('[(*)]', Comment.Multiline)]}