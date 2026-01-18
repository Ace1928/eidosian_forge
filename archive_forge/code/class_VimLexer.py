import re
from bisect import bisect
from pygments.lexer import RegexLexer, include, default, bygroups, using, this
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
class VimLexer(RegexLexer):
    """
    Lexer for VimL script files.

    .. versionadded:: 0.8
    """
    name = 'VimL'
    aliases = ['vim']
    filenames = ['*.vim', '.vimrc', '.exrc', '.gvimrc', '_vimrc', '_exrc', '_gvimrc', 'vimrc', 'gvimrc']
    mimetypes = ['text/x-vim']
    flags = re.MULTILINE
    _python = 'py(?:t(?:h(?:o(?:n)?)?)?)?'
    tokens = {'root': [('^([ \\t:]*)(' + _python + ')([ \\t]*)(<<)([ \\t]*)(.*)((?:\\n|.)*)(\\6)', bygroups(using(this), Keyword, Text, Operator, Text, Text, using(PythonLexer), Text)), ('^([ \\t:]*)(' + _python + ')([ \\t])(.*)', bygroups(using(this), Keyword, Text, using(PythonLexer))), ('^\\s*".*', Comment), ('[ \\t]+', Text), ('/(\\\\\\\\|\\\\/|[^\\n/])*/', String.Regex), ('"(\\\\\\\\|\\\\"|[^\\n"])*"', String.Double), ("'(''|[^\\n'])*'", String.Single), ('(?<=\\s)"[^\\-:.%#=*].*', Comment), ('-?\\d+', Number), ('#[0-9a-f]{6}', Number.Hex), ('^:', Punctuation), ('[()<>+=!|,~-]', Punctuation), ('\\b(let|if|else|endif|elseif|fun|function|endfunction)\\b', Keyword), ('\\b(NONE|bold|italic|underline|dark|light)\\b', Name.Builtin), ('\\b\\w+\\b', Name.Other), ('.', Text)]}

    def __init__(self, **options):
        from pygments.lexers._vim_builtins import command, option, auto
        self._cmd = command
        self._opt = option
        self._aut = auto
        RegexLexer.__init__(self, **options)

    def is_in(self, w, mapping):
        """
        It's kind of difficult to decide if something might be a keyword
        in VimL because it allows you to abbreviate them.  In fact,
        'ab[breviate]' is a good example.  :ab, :abbre, or :abbreviate are
        valid ways to call it so rather than making really awful regexps
        like::

            \\bab(?:b(?:r(?:e(?:v(?:i(?:a(?:t(?:e)?)?)?)?)?)?)?)?\\b

        we match `\\b\\w+\\b` and then call is_in() on those tokens.  See
        `scripts/get_vimkw.py` for how the lists are extracted.
        """
        p = bisect(mapping, (w,))
        if p > 0:
            if mapping[p - 1][0] == w[:len(mapping[p - 1][0])] and mapping[p - 1][1][:len(w)] == w:
                return True
        if p < len(mapping):
            return mapping[p][0] == w[:len(mapping[p][0])] and mapping[p][1][:len(w)] == w
        return False

    def get_tokens_unprocessed(self, text):
        for index, token, value in RegexLexer.get_tokens_unprocessed(self, text):
            if token is Name.Other:
                if self.is_in(value, self._cmd):
                    yield (index, Keyword, value)
                elif self.is_in(value, self._opt) or self.is_in(value, self._aut):
                    yield (index, Name.Builtin, value)
                else:
                    yield (index, Text, value)
            else:
                yield (index, token, value)