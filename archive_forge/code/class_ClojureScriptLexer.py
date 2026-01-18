import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class ClojureScriptLexer(ClojureLexer):
    """
    Lexer for `ClojureScript <http://clojure.org/clojurescript>`_
    source code.

    .. versionadded:: 2.0
    """
    name = 'ClojureScript'
    aliases = ['clojurescript', 'cljs']
    filenames = ['*.cljs']
    mimetypes = ['text/x-clojurescript', 'application/x-clojurescript']