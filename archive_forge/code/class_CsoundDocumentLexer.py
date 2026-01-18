import re
from pygments.lexer import RegexLexer, bygroups, default, include, using, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, \
from pygments.lexers._csound_builtins import OPCODES
from pygments.lexers.html import HtmlLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.scripting import LuaLexer
class CsoundDocumentLexer(RegexLexer):
    """
    For `Csound <http://csound.github.io>`_ documents.

    .. versionadded:: 2.1
    """
    name = 'Csound Document'
    aliases = ['csound-document', 'csound-csd']
    filenames = ['*.csd']
    tokens = {'root': [newline, ('/[*](.|\\n)*?[*]/', Comment.Multiline), ('[^<&;/]+', Text), ('<\\s*CsInstruments', Name.Tag, ('orchestra', 'tag')), ('<\\s*CsScore', Name.Tag, ('score', 'tag')), ('<\\s*[hH][tT][mM][lL]', Name.Tag, ('HTML', 'tag')), ('<\\s*[\\w:.-]+', Name.Tag, 'tag'), ('<\\s*/\\s*[\\w:.-]+\\s*>', Name.Tag)], 'orchestra': [('<\\s*/\\s*CsInstruments\\s*>', Name.Tag, '#pop'), ('(.|\\n)+?(?=<\\s*/\\s*CsInstruments\\s*>)', using(CsoundOrchestraLexer))], 'score': [('<\\s*/\\s*CsScore\\s*>', Name.Tag, '#pop'), ('(.|\\n)+?(?=<\\s*/\\s*CsScore\\s*>)', using(CsoundScoreLexer))], 'HTML': [('<\\s*/\\s*[hH][tT][mM][lL]\\s*>', Name.Tag, '#pop'), ('(.|\\n)+?(?=<\\s*/\\s*[hH][tT][mM][lL]\\s*>)', using(HtmlLexer))], 'tag': [('\\s+', Text), ('[\\w.:-]+\\s*=', Name.Attribute, 'attr'), ('/?\\s*>', Name.Tag, '#pop')], 'attr': [('\\s+', Text), ('".*?"', String, '#pop'), ("'.*?'", String, '#pop'), ('[^\\s>]+', String, '#pop')]}