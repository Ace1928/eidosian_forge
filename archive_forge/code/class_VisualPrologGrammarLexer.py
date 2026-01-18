import re
from pygments.lexer import RegexLexer, inherit, words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class VisualPrologGrammarLexer(VisualPrologBaseLexer):
    """Lexer for VisualProlog grammar

    .. versionadded:: 2.17
    """
    name = 'Visual Prolog Grammar'
    url = 'https://www.visual-prolog.com/'
    aliases = ['visualprologgrammar']
    filenames = ['*.vipgrm']
    majorkw = ('open', 'namespace', 'grammar', 'nonterminals', 'startsymbols', 'terminals', 'rules', 'precedence')
    directivekw = ('bininclude', 'stringinclude')
    tokens = {'root': [(words(majorkw, suffix='\\b'), Keyword), (words(directivekw, prefix='#', suffix='\\b'), Keyword.Directive), inherit]}

    def analyse_text(text):
        """No competditors (currently)"""
        if re.search('^\\s*(end\\s+grammar|(nonterminals|startsymbols|terminals|rules|precedence)\\s*$)', text):
            return 0.98
        else:
            return 0