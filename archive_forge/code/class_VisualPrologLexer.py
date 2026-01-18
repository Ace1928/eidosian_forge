import re
from pygments.lexer import RegexLexer, inherit, words, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class VisualPrologLexer(VisualPrologBaseLexer):
    """Lexer for VisualProlog

    .. versionadded:: 2.17
    """
    name = 'Visual Prolog'
    url = 'https://www.visual-prolog.com/'
    aliases = ['visualprolog']
    filenames = ['*.pro', '*.cl', '*.i', '*.pack', '*.ph']
    majorkw = ('goal', 'namespace', 'interface', 'class', 'implement', 'where', 'open', 'inherits', 'supports', 'resolve', 'delegate', 'monitor', 'constants', 'domains', 'predicates', 'constructors', 'properties', 'clauses', 'facts')
    minorkw = ('align', 'anyflow', 'as', 'bitsize', 'determ', 'digits', 'erroneous', 'externally', 'failure', 'from', 'guard', 'multi', 'nondeterm', 'or', 'orelse', 'otherwise', 'procedure', 'resolve', 'single', 'suspending')
    directivekw = ('bininclude', 'else', 'elseif', 'endif', 'error', 'export', 'externally', 'from', 'grammargenerate', 'grammarinclude', 'if', 'include', 'message', 'options', 'orrequires', 'requires', 'stringinclude', 'then')
    tokens = {'root': [(words(minorkw, suffix='\\b'), Keyword.Minor), (words(majorkw, suffix='\\b'), Keyword), (words(directivekw, prefix='#', suffix='\\b'), Keyword.Directive), inherit]}

    def analyse_text(text):
        """Competes with IDL and Prolog on *.pro; div. lisps on*.cl and SwigLexer on *.i"""
        if re.search('^\\s*(end\\s+(interface|class|implement)|(clauses|predicates|domains|facts|constants|properties)\\s*$)', text):
            return 0.98
        else:
            return 0