import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def _add_syntax_highlighting(self, insource, language):
    lexer = pygments.lexers.get_lexer_by_name(language, stripall=True)
    if language in ('latex', 'tex'):
        fmtr = OdtPygmentsLaTeXFormatter(lambda name, parameters=(): self.rststyle(name, parameters), escape_function=escape_cdata)
    else:
        fmtr = OdtPygmentsProgFormatter(lambda name, parameters=(): self.rststyle(name, parameters), escape_function=escape_cdata)
    outsource = pygments.highlight(insource, lexer, fmtr)
    return outsource