import Fontmapping  # helps by mapping pid font classes to Pyart font names
import pyart
from rdkit.sping.PDF import pdfmetrics
from rdkit.sping.pid import *
def _restoreArtState(self, bool):
    if bool:
        self._pycan.grestore()