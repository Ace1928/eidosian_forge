import os
from ...base import (
class landmarksConstellationAlignerOutputSpec(TraitedSpec):
    outputLandmarksPaired = File(desc='Output landmark file (.fcsv)', exists=True)