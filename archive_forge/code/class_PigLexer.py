import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class PigLexer(RegexLexer):
    """
    For `Pig Latin <https://pig.apache.org/>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'Pig'
    aliases = ['pig']
    filenames = ['*.pig']
    mimetypes = ['text/x-pig']
    flags = re.MULTILINE | re.IGNORECASE
    tokens = {'root': [('\\s+', Text), ('--.*', Comment), ('/\\*[\\w\\W]*?\\*/', Comment.Multiline), ('\\\\\\n', Text), ('\\\\', Text), ("\\'(?:\\\\[ntbrf\\\\\\']|\\\\u[0-9a-f]{4}|[^\\'\\\\\\n\\r])*\\'", String), include('keywords'), include('types'), include('builtins'), include('punct'), include('operators'), ('[0-9]*\\.[0-9]+(e[0-9]+)?[fd]?', Number.Float), ('0x[0-9a-f]+', Number.Hex), ('[0-9]+L?', Number.Integer), ('\\n', Text), ('([a-z_]\\w*)(\\s*)(\\()', bygroups(Name.Function, Text, Punctuation)), ('[()#:]', Text), ('[^(:#\\\'")\\s]+', Text), ('\\S+\\s+', Text)], 'keywords': [('(assert|and|any|all|arrange|as|asc|bag|by|cache|CASE|cat|cd|cp|%declare|%default|define|dense|desc|describe|distinct|du|dump|eval|exex|explain|filter|flatten|foreach|full|generate|group|help|if|illustrate|import|inner|input|into|is|join|kill|left|limit|load|ls|map|matches|mkdir|mv|not|null|onschema|or|order|outer|output|parallel|pig|pwd|quit|register|returns|right|rm|rmf|rollup|run|sample|set|ship|split|stderr|stdin|stdout|store|stream|through|union|using|void)\\b', Keyword)], 'builtins': [('(AVG|BinStorage|cogroup|CONCAT|copyFromLocal|copyToLocal|COUNT|cross|DIFF|MAX|MIN|PigDump|PigStorage|SIZE|SUM|TextLoader|TOKENIZE)\\b', Name.Builtin)], 'types': [('(bytearray|BIGINTEGER|BIGDECIMAL|chararray|datetime|double|float|int|long|tuple)\\b', Keyword.Type)], 'punct': [('[;(){}\\[\\]]', Punctuation)], 'operators': [('[#=,./%+\\-?]', Operator), ('(eq|gt|lt|gte|lte|neq|matches)\\b', Operator), ('(==|<=|<|>=|>|!=)', Operator)]}