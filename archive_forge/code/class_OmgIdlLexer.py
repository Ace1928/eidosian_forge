import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class OmgIdlLexer(CLexer):
    """
    Lexer for Object Management Group Interface Definition Language.

    .. versionadded:: 2.9
    """
    name = 'OMG Interface Definition Language'
    url = 'https://www.omg.org/spec/IDL/About-IDL/'
    aliases = ['omg-idl']
    filenames = ['*.idl', '*.pidl']
    mimetypes = []
    scoped_name = '((::)?\\w+)+'
    tokens = {'values': [(words(('true', 'false'), prefix='(?i)', suffix='\\b'), Number), ('([Ll]?)(")', bygroups(String.Affix, String.Double), 'string'), ("([Ll]?)(\\')(\\\\[^\\']+)(\\')", bygroups(String.Affix, String.Char, String.Escape, String.Char)), ("([Ll]?)(\\')(\\\\\\')(\\')", bygroups(String.Affix, String.Char, String.Escape, String.Char)), ("([Ll]?)(\\'.\\')", bygroups(String.Affix, String.Char)), ('[+-]?\\d+(\\.\\d*)?[Ee][+-]?\\d+', Number.Float), ('[+-]?(\\d+\\.\\d*)|(\\d*\\.\\d+)([Ee][+-]?\\d+)?', Number.Float), ('(?i)[+-]?0x[0-9a-f]+', Number.Hex), ('[+-]?[1-9]\\d*', Number.Integer), ('[+-]?0[0-7]*', Number.Oct), ('[\\+\\-\\*\\/%^&\\|~]', Operator), (words(('<<', '>>')), Operator), (scoped_name, Name), ('[{};:,<>\\[\\]]', Punctuation)], 'annotation_params': [include('whitespace'), ('\\(', Punctuation, '#push'), include('values'), ('=', Punctuation), ('\\)', Punctuation, '#pop')], 'annotation_params_maybe': [('\\(', Punctuation, 'annotation_params'), include('whitespace'), default('#pop')], 'annotation_appl': [('@' + scoped_name, Name.Decorator, 'annotation_params_maybe')], 'enum': [include('whitespace'), ('[{,]', Punctuation), ('\\w+', Name.Constant), include('annotation_appl'), ('\\}', Punctuation, '#pop')], 'root': [include('whitespace'), (words(('typedef', 'const', 'in', 'out', 'inout', 'local'), prefix='(?i)', suffix='\\b'), Keyword.Declaration), (words(('void', 'any', 'native', 'bitfield', 'unsigned', 'boolean', 'char', 'wchar', 'octet', 'short', 'long', 'int8', 'uint8', 'int16', 'int32', 'int64', 'uint16', 'uint32', 'uint64', 'float', 'double', 'fixed', 'sequence', 'string', 'wstring', 'map'), prefix='(?i)', suffix='\\b'), Keyword.Type), (words(('@annotation', 'struct', 'union', 'bitset', 'interface', 'exception', 'valuetype', 'eventtype', 'component'), prefix='(?i)', suffix='(\\s+)(\\w+)'), bygroups(Keyword, Whitespace, Name.Class)), (words(('abstract', 'alias', 'attribute', 'case', 'connector', 'consumes', 'context', 'custom', 'default', 'emits', 'factory', 'finder', 'getraises', 'home', 'import', 'manages', 'mirrorport', 'multiple', 'Object', 'oneway', 'primarykey', 'private', 'port', 'porttype', 'provides', 'public', 'publishes', 'raises', 'readonly', 'setraises', 'supports', 'switch', 'truncatable', 'typeid', 'typename', 'typeprefix', 'uses', 'ValueBase'), prefix='(?i)', suffix='\\b'), Keyword), ('(?i)(enum|bitmask)(\\s+)(\\w+)', bygroups(Keyword, Whitespace, Name.Class), 'enum'), ('(?i)(module)(\\s+)(\\w+)', bygroups(Keyword.Namespace, Whitespace, Name.Namespace)), ('(\\w+)(\\s*)(=)', bygroups(Name.Constant, Whitespace, Operator)), ('[\\(\\)]', Punctuation), include('values'), include('annotation_appl')]}