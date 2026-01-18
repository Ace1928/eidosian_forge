import os
from ...base import (
class landmarksConstellationAligner(SEMLikeCommandLine):
    """title: MidACPC Landmark Insertion

    category: Utilities.BRAINS

    description: This program converts the original landmark files to the acpc-aligned landmark files

    contributor: Ali Ghayoor
    """
    input_spec = landmarksConstellationAlignerInputSpec
    output_spec = landmarksConstellationAlignerOutputSpec
    _cmd = ' landmarksConstellationAligner '
    _outputs_filenames = {'outputLandmarksPaired': 'outputLandmarksPaired'}
    _redirect_x = False