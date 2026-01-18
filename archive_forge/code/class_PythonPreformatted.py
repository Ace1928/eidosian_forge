from reportlab.lib import PyFontify
from reportlab.platypus.paragraph import Paragraph, _handleBulletWidth, \
from reportlab.lib.utils import isSeq
from reportlab.platypus.flowables import _dedenter
class PythonPreformatted(XPreformatted):
    """Used for syntax-colored Python code, otherwise like XPreformatted.
    """
    formats = {'rest': ('', ''), 'comment': ('<font color="green">', '</font>'), 'keyword': ('<font color="blue"><b>', '</b></font>'), 'parameter': ('<font color="black">', '</font>'), 'identifier': ('<font color="red">', '</font>'), 'string': ('<font color="gray">', '</font>')}

    def __init__(self, text, style, bulletText=None, dedent=0, frags=None):
        if text:
            text = self.fontify(self.escapeHtml(text))
        XPreformatted.__init__(self, text, style, bulletText=bulletText, dedent=dedent, frags=frags)

    def escapeHtml(self, text):
        s = text.replace('&', '&amp;')
        s = s.replace('<', '&lt;')
        s = s.replace('>', '&gt;')
        return s

    def fontify(self, code):
        """Return a fontified version of some Python code."""
        if code[0] == '\n':
            code = code[1:]
        tags = PyFontify.fontify(code)
        fontifiedCode = ''
        pos = 0
        for k, i, j, dummy in tags:
            fontifiedCode = fontifiedCode + code[pos:i]
            s, e = self.formats[k]
            fontifiedCode = fontifiedCode + s + code[i:j] + e
            pos = j
        fontifiedCode = fontifiedCode + code[pos:]
        return fontifiedCode