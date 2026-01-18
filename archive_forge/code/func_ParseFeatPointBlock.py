import re
from rdkit import Geometry
from rdkit.Chem.FeatMaps import FeatMapPoint, FeatMaps
def ParseFeatPointBlock(self):
    featLineSplitter = re.compile('([a-zA-Z]+) *= *')
    feats = []
    l = self._NextLine()
    while l and l != 'EndPoints':
        vals = featLineSplitter.split(l)
        while vals.count(''):
            vals.remove('')
        p = FeatMapPoint.FeatMapPoint()
        for i in range(0, len(vals), 2):
            name = vals[i].lower()
            value = vals[i + 1]
            if name == 'family':
                p.SetFamily(value.strip())
            elif name == 'weight':
                p.weight = float(value)
            elif name == 'pos':
                p.SetPos(self._parsePoint(value))
            elif name == 'dir':
                p.featDirs.append(self._parsePoint(value))
            else:
                raise FeatMapParseError(f'FeatPoint option {name} not recognized on line {self._lineNum}')
        feats.append(p)
        l = self._NextLine()
    return feats