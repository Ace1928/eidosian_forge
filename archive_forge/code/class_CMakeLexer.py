import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class CMakeLexer(RegexLexer):
    """
    Lexer for `CMake <http://cmake.org/Wiki/CMake>`_ files.

    .. versionadded:: 1.2
    """
    name = 'CMake'
    aliases = ['cmake']
    filenames = ['*.cmake', 'CMakeLists.txt']
    mimetypes = ['text/x-cmake']
    tokens = {'root': [('\\b(\\w+)([ \\t]*)(\\()', bygroups(Name.Builtin, Text, Punctuation), 'args'), include('keywords'), include('ws')], 'args': [('\\(', Punctuation, '#push'), ('\\)', Punctuation, '#pop'), ('(\\$\\{)(.+?)(\\})', bygroups(Operator, Name.Variable, Operator)), ('(\\$ENV\\{)(.+?)(\\})', bygroups(Operator, Name.Variable, Operator)), ('(\\$<)(.+?)(>)', bygroups(Operator, Name.Variable, Operator)), ('(?s)".*?"', String.Double), ('\\\\\\S+', String), ('[^)$"# \\t\\n]+', String), ('\\n', Text), include('keywords'), include('ws')], 'string': [], 'keywords': [('\\b(WIN32|UNIX|APPLE|CYGWIN|BORLAND|MINGW|MSVC|MSVC_IDE|MSVC60|MSVC70|MSVC71|MSVC80|MSVC90)\\b', Keyword)], 'ws': [('[ \\t]+', Text), ('#.*\\n', Comment)]}

    def analyse_text(text):
        exp = '^ *CMAKE_MINIMUM_REQUIRED *\\( *VERSION *\\d(\\.\\d)* *( FATAL_ERROR)? *\\) *$'
        if re.search(exp, text, flags=re.MULTILINE | re.IGNORECASE):
            return 0.8
        return 0.0