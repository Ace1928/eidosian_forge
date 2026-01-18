import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class LlvmMirLexer(RegexLexer):
    """
    Lexer for the overall LLVM MIR document format.

    MIR is a human readable serialization format that's used to represent LLVM's
    machine specific intermediate representation. It allows LLVM's developers to
    see the state of the compilation process at various points, as well as test
    individual pieces of the compiler.

    .. versionadded:: 2.6
    """
    name = 'LLVM-MIR'
    url = 'https://llvm.org/docs/MIRLangRef.html'
    aliases = ['llvm-mir']
    filenames = ['*.mir']
    tokens = {'root': [('#.*', Comment), ('--- \\|$', Keyword, 'llvm_ir'), ('---', Keyword, 'llvm_mir'), ('[^-#]+|.', Text)], 'llvm_ir': [('(\\.\\.\\.|(?=---))', Keyword, '#pop'), ('((?:.|\\n)+?)(?=(\\.\\.\\.|---))', bygroups(using(LlvmLexer)))], 'llvm_mir': [('#.*', Comment), ('(\\.\\.\\.|(?=---))', Keyword, '#pop'), ('name:', Keyword, 'name'), (words(('alignment',), suffix=':'), Keyword, 'number'), (words(('legalized', 'regBankSelected', 'tracksRegLiveness', 'selected', 'exposesReturnsTwice'), suffix=':'), Keyword, 'boolean'), (words(('registers', 'stack', 'fixedStack', 'liveins', 'frameInfo', 'machineFunctionInfo'), suffix=':'), Keyword), ('body: *\\|', Keyword, 'llvm_mir_body'), ('.+', Text), ('\\n', Whitespace)], 'name': [('[^\\n]+', Name), default('#pop')], 'boolean': [(' *(true|false)', Name.Builtin), default('#pop')], 'number': [(' *[0-9]+', Number), default('#pop')], 'llvm_mir_body': [('(\\.\\.\\.|(?=---))', Keyword, '#pop:2'), ('((?:.|\\n)+?)(?=\\.\\.\\.|---)', bygroups(using(LlvmMirBodyLexer))), ('(?!\\.\\.\\.|---)((?:.|\\n)+)', bygroups(using(LlvmMirBodyLexer)))]}