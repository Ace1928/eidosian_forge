import re
from pygments.lexer import RegexLexer, DelegatingLexer, \
from pygments.token import Punctuation, Other, Text, Comment, Operator, \
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers.objective import ObjectiveCLexer
from pygments.lexers.d import DLexer
from pygments.lexers.dotnet import CSharpLexer
from pygments.lexers.ruby import RubyLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
class TreetopBaseLexer(RegexLexer):
    """
    A base lexer for `Treetop <http://treetop.rubyforge.org/>`_ grammars.
    Not for direct use; use TreetopLexer instead.

    .. versionadded:: 1.6
    """
    tokens = {'root': [include('space'), ('require[ \\t]+[^\\n\\r]+[\\n\\r]', Other), ('module\\b', Keyword.Namespace, 'module'), ('grammar\\b', Keyword, 'grammar')], 'module': [include('space'), include('end'), ('module\\b', Keyword, '#push'), ('grammar\\b', Keyword, 'grammar'), ('[A-Z]\\w*(?:::[A-Z]\\w*)*', Name.Namespace)], 'grammar': [include('space'), include('end'), ('rule\\b', Keyword, 'rule'), ('include\\b', Keyword, 'include'), ('[A-Z]\\w*', Name)], 'include': [include('space'), ('[A-Z]\\w*(?:::[A-Z]\\w*)*', Name.Class, '#pop')], 'rule': [include('space'), include('end'), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('([A-Za-z_]\\w*)(:)', bygroups(Name.Label, Punctuation)), ('[A-Za-z_]\\w*', Name), ('[()]', Punctuation), ('[?+*/&!~]', Operator), ('\\[(?:\\\\.|\\[:\\^?[a-z]+:\\]|[^\\\\\\]])+\\]', String.Regex), ('([0-9]*)(\\.\\.)([0-9]*)', bygroups(Number.Integer, Operator, Number.Integer)), ('(<)([^>]+)(>)', bygroups(Punctuation, Name.Class, Punctuation)), ('\\{', Punctuation, 'inline_module'), ('\\.', String.Regex)], 'inline_module': [('\\{', Other, 'ruby'), ('\\}', Punctuation, '#pop'), ('[^{}]+', Other)], 'ruby': [('\\{', Other, '#push'), ('\\}', Other, '#pop'), ('[^{}]+', Other)], 'space': [('[ \\t\\n\\r]+', Whitespace), ('#[^\\n]*', Comment.Single)], 'end': [('end\\b', Keyword, '#pop')]}