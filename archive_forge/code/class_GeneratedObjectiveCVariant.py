import re
from pygments.lexer import RegexLexer, include, bygroups, using, this, words, \
from pygments.token import Text, Keyword, Name, String, Operator, \
from pygments.lexers.c_cpp import CLexer, CppLexer
class GeneratedObjectiveCVariant(baselexer):
    """
        Implements Objective-C syntax on top of an existing C family lexer.
        """
    tokens = {'statements': [('@"', String, 'string'), ('@(YES|NO)', Number), ("@'(\\\\.|\\\\[0-7]{1,3}|\\\\x[a-fA-F0-9]{1,2}|[^\\\\\\'\\n])'", String.Char), ('@(\\d+\\.\\d*|\\.\\d+|\\d+)[eE][+-]?\\d+[lL]?', Number.Float), ('@(\\d+\\.\\d*|\\.\\d+|\\d+[fF])[fF]?', Number.Float), ('@0x[0-9a-fA-F]+[Ll]?', Number.Hex), ('@0[0-7]+[Ll]?', Number.Oct), ('@\\d+[Ll]?', Number.Integer), ('@\\(', Literal, 'literal_number'), ('@\\[', Literal, 'literal_array'), ('@\\{', Literal, 'literal_dictionary'), (words(('@selector', '@private', '@protected', '@public', '@encode', '@synchronized', '@try', '@throw', '@catch', '@finally', '@end', '@property', '@synthesize', '__bridge', '__bridge_transfer', '__autoreleasing', '__block', '__weak', '__strong', 'weak', 'strong', 'copy', 'retain', 'assign', 'unsafe_unretained', 'atomic', 'nonatomic', 'readonly', 'readwrite', 'setter', 'getter', 'typeof', 'in', 'out', 'inout', 'release', 'class', '@dynamic', '@optional', '@required', '@autoreleasepool'), suffix='\\b'), Keyword), (words(('id', 'instancetype', 'Class', 'IMP', 'SEL', 'BOOL', 'IBOutlet', 'IBAction', 'unichar'), suffix='\\b'), Keyword.Type), ('@(true|false|YES|NO)\\n', Name.Builtin), ('(YES|NO|nil|self|super)\\b', Name.Builtin), ('(Boolean|UInt8|SInt8|UInt16|SInt16|UInt32|SInt32)\\b', Keyword.Type), ('(TRUE|FALSE)\\b', Name.Builtin), ('(@interface|@implementation)(\\s+)', bygroups(Keyword, Text), ('#pop', 'oc_classname')), ('(@class|@protocol)(\\s+)', bygroups(Keyword, Text), ('#pop', 'oc_forward_classname')), ('@', Punctuation), inherit], 'oc_classname': [('([a-zA-Z$_][\\w$]*)(\\s*:\\s*)([a-zA-Z$_][\\w$]*)?(\\s*)(\\{)', bygroups(Name.Class, Text, Name.Class, Text, Punctuation), ('#pop', 'oc_ivars')), ('([a-zA-Z$_][\\w$]*)(\\s*:\\s*)([a-zA-Z$_][\\w$]*)?', bygroups(Name.Class, Text, Name.Class), '#pop'), ('([a-zA-Z$_][\\w$]*)(\\s*)(\\([a-zA-Z$_][\\w$]*\\))(\\s*)(\\{)', bygroups(Name.Class, Text, Name.Label, Text, Punctuation), ('#pop', 'oc_ivars')), ('([a-zA-Z$_][\\w$]*)(\\s*)(\\([a-zA-Z$_][\\w$]*\\))', bygroups(Name.Class, Text, Name.Label), '#pop'), ('([a-zA-Z$_][\\w$]*)(\\s*)(\\{)', bygroups(Name.Class, Text, Punctuation), ('#pop', 'oc_ivars')), ('([a-zA-Z$_][\\w$]*)', Name.Class, '#pop')], 'oc_forward_classname': [('([a-zA-Z$_][\\w$]*)(\\s*,\\s*)', bygroups(Name.Class, Text), 'oc_forward_classname'), ('([a-zA-Z$_][\\w$]*)(\\s*;?)', bygroups(Name.Class, Text), '#pop')], 'oc_ivars': [include('whitespace'), include('statements'), (';', Punctuation), ('\\{', Punctuation, '#push'), ('\\}', Punctuation, '#pop')], 'root': [('^([-+])(\\s*)(\\(.*?\\))?(\\s*)([a-zA-Z$_][\\w$]*:?)', bygroups(Punctuation, Text, using(this), Text, Name.Function), 'method'), inherit], 'method': [include('whitespace'), (',', Punctuation), ('\\.\\.\\.', Punctuation), ('(\\(.*?\\))(\\s*)([a-zA-Z$_][\\w$]*)', bygroups(using(this), Text, Name.Variable)), ('[a-zA-Z$_][\\w$]*:', Name.Function), (';', Punctuation, '#pop'), ('\\{', Punctuation, 'function'), default('#pop')], 'literal_number': [('\\(', Punctuation, 'literal_number_inner'), ('\\)', Literal, '#pop'), include('statement')], 'literal_number_inner': [('\\(', Punctuation, '#push'), ('\\)', Punctuation, '#pop'), include('statement')], 'literal_array': [('\\[', Punctuation, 'literal_array_inner'), ('\\]', Literal, '#pop'), include('statement')], 'literal_array_inner': [('\\[', Punctuation, '#push'), ('\\]', Punctuation, '#pop'), include('statement')], 'literal_dictionary': [('\\}', Literal, '#pop'), include('statement')]}

    def analyse_text(text):
        if _oc_keywords.search(text):
            return 1.0
        elif '@"' in text:
            return 0.8
        elif re.search('@[0-9]+', text):
            return 0.7
        elif _oc_message.search(text):
            return 0.8
        return 0

    def get_tokens_unprocessed(self, text):
        from pygments.lexers._cocoa_builtins import COCOA_INTERFACES, COCOA_PROTOCOLS, COCOA_PRIMITIVES
        for index, token, value in baselexer.get_tokens_unprocessed(self, text):
            if token is Name or token is Name.Class:
                if value in COCOA_INTERFACES or value in COCOA_PROTOCOLS or value in COCOA_PRIMITIVES:
                    token = Name.Builtin.Pseudo
            yield (index, token, value)