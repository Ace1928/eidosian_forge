from fontTools.misc import psCharStrings
from fontTools import ttLib
from fontTools.pens.basePen import NullPen
from fontTools.misc.roundTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.subset.util import _add_method, _uniq_sort
def _empty_charstring(font, glyphName, isCFF2, ignoreWidth=False):
    c, fdSelectIndex = font.CharStrings.getItemAndSelector(glyphName)
    if isCFF2 or ignoreWidth:
        c.setProgram([] if isCFF2 else ['endchar'])
    else:
        if hasattr(font, 'FDArray') and font.FDArray is not None:
            private = font.FDArray[fdSelectIndex].Private
        else:
            private = font.Private
        dfltWdX = private.defaultWidthX
        nmnlWdX = private.nominalWidthX
        pen = NullPen()
        c.draw(pen)
        if c.width != dfltWdX:
            c.program = [c.width - nmnlWdX, 'endchar']
        else:
            c.program = ['endchar']