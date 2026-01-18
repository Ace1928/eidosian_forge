from pygments.lexer import RegexLexer, words
from pygments.token import Comment, Keyword, Name, String, Number, \
class TalLexer(RegexLexer):
    """
    For `Uxntal <https://wiki.xxiivv.com/site/uxntal.html>`_ source code.

    .. versionadded:: 2.12
    """
    name = 'Tal'
    aliases = ['tal', 'uxntal']
    filenames = ['*.tal']
    mimetypes = ['text/x-uxntal']
    instructions = ['BRK', 'LIT', 'INC', 'POP', 'DUP', 'NIP', 'SWP', 'OVR', 'ROT', 'EQU', 'NEQ', 'GTH', 'LTH', 'JMP', 'JCN', 'JSR', 'STH', 'LDZ', 'STZ', 'LDR', 'STR', 'LDA', 'STA', 'DEI', 'DEO', 'ADD', 'SUB', 'MUL', 'DIV', 'AND', 'ORA', 'EOR', 'SFT']
    tokens = {'comment': [('(?<!\\S)\\((?!\\S)', Comment.Multiline, '#push'), ('(?<!\\S)\\)(?!\\S)', Comment.Multiline, '#pop'), ('[^()]+', Comment.Multiline), ('[()]+', Comment.Multiline)], 'root': [('\\s+', Whitespace), ('(?<!\\S)\\((?!\\S)', Comment.Multiline, 'comment'), (words(instructions, prefix='(?<!\\S)', suffix='2?k?r?(?!\\S)'), Keyword.Reserved), ('[][{}](?!\\S)', Punctuation), ('#([0-9a-f]{2}){1,2}(?!\\S)', Number.Hex), ('"\\S+', String), ('([0-9a-f]{2}){1,2}(?!\\S)', Literal), ('[|$][0-9a-f]{1,4}(?!\\S)', Keyword.Declaration), ('%\\S+', Name.Decorator), ('@\\S+', Name.Function), ('&\\S+', Name.Label), ('/\\S+', Name.Tag), ('\\.\\S+', Name.Variable.Magic), (',\\S+', Name.Variable.Instance), (';\\S+', Name.Variable.Global), ('-\\S+', Literal), ('_\\S+', Literal), ('=\\S+', Literal), ('!\\S+', Name.Function), ('\\?\\S+', Name.Function), ('~\\S+', Keyword.Namespace), ('\\S+', Name.Function)]}

    def analyse_text(text):
        return '|0100' in text[:500]