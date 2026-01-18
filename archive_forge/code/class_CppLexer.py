import re
from pygments.lexer import RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CppLexer(CFamilyLexer):
    """
    For C++ source code with preprocessor directives.
    """
    name = 'C++'
    aliases = ['cpp', 'c++']
    filenames = ['*.cpp', '*.hpp', '*.c++', '*.h++', '*.cc', '*.hh', '*.cxx', '*.hxx', '*.C', '*.H', '*.cp', '*.CPP']
    mimetypes = ['text/x-c++hdr', 'text/x-c++src']
    priority = 0.1
    tokens = {'statements': [(words(('catch', 'const_cast', 'delete', 'dynamic_cast', 'explicit', 'export', 'friend', 'mutable', 'namespace', 'new', 'operator', 'private', 'protected', 'public', 'reinterpret_cast', 'restrict', 'static_cast', 'template', 'this', 'throw', 'throws', 'try', 'typeid', 'typename', 'using', 'virtual', 'constexpr', 'nullptr', 'decltype', 'thread_local', 'alignas', 'alignof', 'static_assert', 'noexcept', 'override', 'final'), suffix='\\b'), Keyword), ('char(16_t|32_t)\\b', Keyword.Type), ('(class)(\\s+)', bygroups(Keyword, Text), 'classname'), ('(R)(")([^\\\\()\\s]{,16})(\\()((?:.|\\n)*?)(\\)\\3)(")', bygroups(String.Affix, String, String.Delimiter, String.Delimiter, String, String.Delimiter, String)), ('(u8|u|U)(")', bygroups(String.Affix, String), 'string'), inherit], 'root': [inherit, (words(('virtual_inheritance', 'uuidof', 'super', 'single_inheritance', 'multiple_inheritance', 'interface', 'event'), prefix='__', suffix='\\b'), Keyword.Reserved), ('__(offload|blockingoffload|outer)\\b', Keyword.Pseudo)], 'classname': [('[a-zA-Z_]\\w*', Name.Class, '#pop'), ('\\s*(?=>)', Text, '#pop')]}

    def analyse_text(text):
        if re.search('#include <[a-z_]+>', text):
            return 0.2
        if re.search('using namespace ', text):
            return 0.4