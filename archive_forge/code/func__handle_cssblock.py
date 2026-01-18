import re
from pygments.lexers.html import XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.lilypond import LilyPondLexer
from pygments.lexers.data import JsonLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
def _handle_cssblock(self, match):
    """
        match args: 1:style tag 2:newline, 3:code, 4:closing style tag
        """
    from pygments.lexers import get_lexer_by_name
    yield (match.start(1), String, match.group(1))
    yield (match.start(2), String, match.group(2))
    lexer = None
    if self.handlecodeblocks:
        try:
            lexer = get_lexer_by_name('css')
        except ClassNotFound:
            pass
    code = match.group(3)
    if lexer is None:
        yield (match.start(3), String, code)
        return
    yield from do_insertions([], lexer.get_tokens_unprocessed(code))
    yield (match.start(4), String, match.group(4))