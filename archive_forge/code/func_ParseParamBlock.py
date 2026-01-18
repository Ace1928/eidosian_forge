import re
from rdkit import Geometry
from rdkit.Chem.FeatMaps import FeatMapPoint, FeatMaps
def ParseParamBlock(self):
    paramLineSplitter = re.compile('([a-zA-Z]+) *= *(\\S+)')
    params = {}
    l = self._NextLine()
    while l and l != 'EndParams':
        param = FeatMaps.FeatMapParams()
        vals = paramLineSplitter.findall(l)
        for name, val in vals:
            name = name.lower()
            if name == 'family':
                family = val
            elif name == 'radius':
                param.radius = float(val)
            elif name == 'width':
                param.width = float(val)
            elif name == 'profile':
                try:
                    param.featProfile = getattr(param.FeatProfile, val)
                except AttributeError:
                    raise FeatMapParseError('Profile %s not recognized on line %d' % (val, self._lineNum))
            else:
                raise FeatMapParseError('FeatMapParam option %s not recognized on line %d' % (name, self._lineNum))
        params[family] = param
        l = self._NextLine()
    if l != 'EndParams':
        raise FeatMapParseError('EndParams line not found')
    return params