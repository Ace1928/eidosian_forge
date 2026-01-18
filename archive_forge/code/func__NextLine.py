import re
from rdkit import Geometry
from rdkit.Chem.FeatMaps import FeatMapPoint, FeatMaps
def _NextLine(self):
    txt = ''
    while 1:
        try:
            l = self.data[self._lineNum].split('#')[0].strip()
        except IndexError:
            break
        self._lineNum += 1
        if l:
            txt += l
            if l[-1] != '\\':
                break
    return txt