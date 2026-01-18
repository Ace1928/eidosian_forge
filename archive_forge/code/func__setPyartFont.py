import Fontmapping  # helps by mapping pid font classes to Pyart font names
import pyart
from rdkit.sping.PDF import pdfmetrics
from rdkit.sping.pid import *
def _setPyartFont(self, fontInstance):
    fontsize = fontInstance.size
    self._pycan.gstate.font_size = fontsize
    pyartname = Fontmapping.getPyartName(fontInstance)
    self._pycan.gstate.setfont(pyartname)