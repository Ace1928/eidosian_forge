import re
from pygments.lexer import RegexLexer, include, bygroups, using, this, words, \
from pygments.token import Text, Keyword, Name, String, Operator, \
from pygments.lexers.c_cpp import CLexer, CppLexer
class ObjectiveCLexer(objective(CLexer)):
    """
    For Objective-C source code with preprocessor directives.
    """
    name = 'Objective-C'
    aliases = ['objective-c', 'objectivec', 'obj-c', 'objc']
    filenames = ['*.m', '*.h']
    mimetypes = ['text/x-objective-c']
    priority = 0.05