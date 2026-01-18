from reportlab.platypus.flowables import Flowable, Preformatted
from reportlab import rl_config
from reportlab.lib.styles import PropertySet, ParagraphStyle, _baseFontName
from reportlab.lib import colors
from reportlab.lib.utils import annotateException, IdentStr, flatten, isStr, asNative, strTypes, __UNSET__
from reportlab.lib.validators import isListOfNumbersOrNone
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.abag import ABag as CellFrame
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus.doctemplate import Indenter, NullActionFlowable
from reportlab.platypus.flowables import LIIndenter
from collections import namedtuple
class TableRenderCB:
    """table render callback abstract base klass to be called in Table.draw"""

    def __call__(self, T, cmd, *args):
        if not isinstance(T, Table):
            raise ValueError(f'TableRenderCB first argument, {repr(T)} is not a Table')
        meth = getattr(self, cmd, None)
        if not meth:
            raise ValueError(f'invalid TablerenderCB cmd {cmd}')
        meth(T, *args)

    def startTable(self, T):
        raise NotImplementedError('startTable')

    def startBG(self, T):
        raise NotImplementedError('startBG')

    def endBG(self, T):
        raise NotImplementedError('endBG')

    def startRow(self, T, rowNo):
        raise NotImplementedError('startRow')

    def startCell(self, T, rowNo, colNo, cellval, cellstyle, pos, size):
        raise NotImplementedError('startCell')

    def endCell(self, T):
        raise NotImplementedError('endCell')

    def endRow(self, T):
        raise NotImplementedError('endRow')

    def startLines(self, T):
        raise NotImplementedError('startLines')

    def endLines(self, T):
        raise NotImplementedError('endLines')

    def endTable(self, T):
        raise NotImplementedError('endTable')