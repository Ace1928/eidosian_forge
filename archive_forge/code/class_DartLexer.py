import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, iteritems
import pygments.unistring as uni
class DartLexer(RegexLexer):
    """
    For `Dart <http://dartlang.org/>`_ source code.

    .. versionadded:: 1.5
    """
    name = 'Dart'
    aliases = ['dart']
    filenames = ['*.dart']
    mimetypes = ['text/x-dart']
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [include('string_literal'), ('#!(.*?)$', Comment.Preproc), ('\\b(import|export)\\b', Keyword, 'import_decl'), ('\\b(library|source|part of|part)\\b', Keyword), ('[^\\S\\n]+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('\\b(class)\\b(\\s+)', bygroups(Keyword.Declaration, Text), 'class'), ('\\b(assert|break|case|catch|continue|default|do|else|finally|for|if|in|is|new|return|super|switch|this|throw|try|while)\\b', Keyword), ('\\b(abstract|async|await|const|extends|factory|final|get|implements|native|operator|set|static|sync|typedef|var|with|yield)\\b', Keyword.Declaration), ('\\b(bool|double|dynamic|int|num|Object|String|void)\\b', Keyword.Type), ('\\b(false|null|true)\\b', Keyword.Constant), ('[~!%^&*+=|?:<>/-]|as\\b', Operator), ('[a-zA-Z_$]\\w*:', Name.Label), ('[a-zA-Z_$]\\w*', Name), ('[(){}\\[\\],.;]', Punctuation), ('0[xX][0-9a-fA-F]+', Number.Hex), ('\\d+(\\.\\d*)?([eE][+-]?\\d+)?', Number), ('\\.\\d+([eE][+-]?\\d+)?', Number), ('\\n', Text)], 'class': [('[a-zA-Z_$]\\w*', Name.Class, '#pop')], 'import_decl': [include('string_literal'), ('\\s+', Text), ('\\b(as|show|hide)\\b', Keyword), ('[a-zA-Z_$]\\w*', Name), ('\\,', Punctuation), ('\\;', Punctuation, '#pop')], 'string_literal': [('r"""([\\w\\W]*?)"""', String.Double), ("r'''([\\w\\W]*?)'''", String.Single), ('r"(.*?)"', String.Double), ("r'(.*?)'", String.Single), ('"""', String.Double, 'string_double_multiline'), ("'''", String.Single, 'string_single_multiline'), ('"', String.Double, 'string_double'), ("'", String.Single, 'string_single')], 'string_common': [('\\\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|u\\{[0-9A-Fa-f]*\\}|[a-z\'\\"$\\\\])', String.Escape), ('(\\$)([a-zA-Z_]\\w*)', bygroups(String.Interpol, Name)), ('(\\$\\{)(.*?)(\\})', bygroups(String.Interpol, using(this), String.Interpol))], 'string_double': [('"', String.Double, '#pop'), ('[^"$\\\\\\n]+', String.Double), include('string_common'), ('\\$+', String.Double)], 'string_double_multiline': [('"""', String.Double, '#pop'), ('[^"$\\\\]+', String.Double), include('string_common'), ('(\\$|\\")+', String.Double)], 'string_single': [("'", String.Single, '#pop'), ("[^'$\\\\\\n]+", String.Single), include('string_common'), ('\\$+', String.Single)], 'string_single_multiline': [("'''", String.Single, '#pop'), ("[^\\'$\\\\]+", String.Single), include('string_common'), ("(\\$|\\')+", String.Single)]}