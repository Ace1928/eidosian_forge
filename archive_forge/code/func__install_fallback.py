import sys
import traceback
from mako import compat
from mako import util
def _install_fallback():
    global syntax_highlight, pygments_html_formatter
    from mako.filters import html_escape
    pygments_html_formatter = None

    def syntax_highlight(filename='', language=None):
        return html_escape