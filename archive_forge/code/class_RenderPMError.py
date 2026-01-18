from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import open_and_read, isBytes, rl_exec
from .shapes import _baseGFontName, _PATH_OP_ARG_COUNT, _PATH_OP_NAMES, definePath
from sys import exc_info
class RenderPMError(Exception):
    pass