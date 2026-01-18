import os
import os.path
import re
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.writers import latex2e
class XeLaTeXTranslator(latex2e.LaTeXTranslator):
    """
    Generate code for LaTeX using Unicode fonts (XeLaTex or LuaLaTeX).

    See the docstring of docutils.writers._html_base.HTMLTranslator for
    notes on and examples of safe subclassing.
    """

    def __init__(self, document):
        self.is_xetex = True
        latex2e.LaTeXTranslator.__init__(self, document, Babel)
        if self.latex_encoding == 'utf8':
            self.requirements.pop('_inputenc', None)
        else:
            self.requirements['_inputenc'] = '\\XeTeXinputencoding %s ' % self.latex_encoding