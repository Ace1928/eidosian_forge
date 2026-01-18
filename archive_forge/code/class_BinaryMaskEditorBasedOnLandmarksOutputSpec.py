import os
from ...base import (
class BinaryMaskEditorBasedOnLandmarksOutputSpec(TraitedSpec):
    outputBinaryVolume = File(desc='Output binary image in which to be edited', exists=True)