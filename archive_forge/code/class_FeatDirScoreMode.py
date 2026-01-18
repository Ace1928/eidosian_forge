import math
from rdkit.Chem.FeatMaps.FeatMapPoint import FeatMapPoint
class FeatDirScoreMode(object):
    Ignore = 0
    ' ignore feature directions\n  '
    DotFullRange = 1
    ' Use the dot product and allow negative contributions when\n      directions are anti-parallel.\n      e.g. score = dot(f1Dir,f2Dir)\n  '
    DotPosRange = 2
    ' Use the dot product and scale contributions to lie between\n      zero and one.\n      e.g. score = ( dot(f1Dir,f2Dir) + 1 ) / 2\n  '