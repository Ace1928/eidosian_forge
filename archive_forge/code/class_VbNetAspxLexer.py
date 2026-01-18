import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer
class VbNetAspxLexer(DelegatingLexer):
    """
    Lexer for highlighting Visual Basic.net within ASP.NET pages.
    """
    name = 'aspx-vb'
    aliases = ['aspx-vb']
    filenames = ['*.aspx', '*.asax', '*.ascx', '*.ashx', '*.asmx', '*.axd']
    mimetypes = []

    def __init__(self, **options):
        super(VbNetAspxLexer, self).__init__(VbNetLexer, GenericAspxLexer, **options)

    def analyse_text(text):
        if re.search('Page\\s*Language="Vb"', text, re.I) is not None:
            return 0.2
        elif re.search('script[^>]+language=["\\\']vb', text, re.I) is not None:
            return 0.15