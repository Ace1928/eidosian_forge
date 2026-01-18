import re
from rdkit import Geometry
from rdkit.Chem.FeatMaps import FeatMapPoint, FeatMaps
class FeatMapParseError(ValueError):
    pass