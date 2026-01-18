import re
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.lexer import RegexLexer, bygroups, default, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
class NotmuchLexer(RegexLexer):
    """
    For Notmuch email text format.

    .. versionadded:: 2.5

    Additional options accepted:

    `body_lexer`
        If given, highlight the contents of the message body with the specified
        lexer, else guess it according to the body content (default: ``None``).
    """
    name = 'Notmuch'
    url = 'https://notmuchmail.org/'
    aliases = ['notmuch']

    def _highlight_code(self, match):
        code = match.group(1)
        try:
            if self.body_lexer:
                lexer = get_lexer_by_name(self.body_lexer)
            else:
                lexer = guess_lexer(code.strip())
        except ClassNotFound:
            lexer = get_lexer_by_name('text')
        yield from lexer.get_tokens_unprocessed(code)
    tokens = {'root': [('\\fmessage\\{\\s*', Keyword, ('message', 'message-attr'))], 'message-attr': [('(\\s*id:\\s*)(\\S+)', bygroups(Name.Attribute, String)), ('(\\s*(?:depth|match|excluded):\\s*)(\\d+)', bygroups(Name.Attribute, Number.Integer)), ('(\\s*filename:\\s*)(.+\\n)', bygroups(Name.Attribute, String)), default('#pop')], 'message': [('\\fmessage\\}\\n', Keyword, '#pop'), ('\\fheader\\{\\n', Keyword, 'header'), ('\\fbody\\{\\n', Keyword, 'body')], 'header': [('\\fheader\\}\\n', Keyword, '#pop'), ('((?:Subject|From|To|Cc|Date):\\s*)(.*\\n)', bygroups(Name.Attribute, String)), ('(.*)(\\s*\\(.*\\))(\\s*\\(.*\\)\\n)', bygroups(Generic.Strong, Literal, Name.Tag))], 'body': [('\\fpart\\{\\n', Keyword, 'part'), ('\\f(part|attachment)\\{\\s*', Keyword, ('part', 'part-attr')), ('\\fbody\\}\\n', Keyword, '#pop')], 'part-attr': [('(ID:\\s*)(\\d+)', bygroups(Name.Attribute, Number.Integer)), ('(,\\s*)((?:Filename|Content-id):\\s*)([^,]+)', bygroups(Punctuation, Name.Attribute, String)), ('(,\\s*)(Content-type:\\s*)(.+\\n)', bygroups(Punctuation, Name.Attribute, String)), default('#pop')], 'part': [('\\f(?:part|attachment)\\}\\n', Keyword, '#pop'), ('\\f(?:part|attachment)\\{\\s*', Keyword, ('#push', 'part-attr')), ('^Non-text part: .*\\n', Comment), ('(?s)(.*?(?=\\f(?:part|attachment)\\}\\n))', _highlight_code)]}

    def analyse_text(text):
        return 1.0 if text.startswith('\x0cmessage{') else 0.0

    def __init__(self, **options):
        self.body_lexer = options.get('body_lexer', None)
        RegexLexer.__init__(self, **options)