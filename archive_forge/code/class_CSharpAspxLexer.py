import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer
class CSharpAspxLexer(DelegatingLexer):
    """
    Lexer for highlighting C# within ASP.NET pages.
    """
    name = 'aspx-cs'
    aliases = ['aspx-cs']
    filenames = ['*.aspx', '*.asax', '*.ascx', '*.ashx', '*.asmx', '*.axd']
    mimetypes = []

    def __init__(self, **options):
        super(CSharpAspxLexer, self).__init__(CSharpLexer, GenericAspxLexer, **options)

    def analyse_text(text):
        if re.search('Page\\s*Language="C#"', text, re.I) is not None:
            return 0.2
        elif re.search('script[^>]+language=["\\\']C#', text, re.I) is not None:
            return 0.15