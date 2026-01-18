from __future__ import absolute_import
import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
from . import Version
from .Code import CCodeWriter
from .. import Utils
def _htmlify_code(self, code, language):
    try:
        from pygments import highlight
        from pygments.lexers import CythonLexer, CppLexer
        from pygments.formatters import HtmlFormatter
    except ImportError:
        return html_escape(code)
    if language == 'cython':
        lexer = CythonLexer(stripnl=False, stripall=False)
    elif language == 'c/cpp':
        lexer = CppLexer(stripnl=False, stripall=False)
    else:
        return html_escape(code)
    html_code = highlight(code, lexer, HtmlFormatter(nowrap=True))
    return html_code