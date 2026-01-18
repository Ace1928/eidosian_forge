import os
from ...base import (
class landmarksConstellationWeights(SEMLikeCommandLine):
    """title: Generate Landmarks Weights (BRAINS)

    category: Utilities.BRAINS

    description: Train up a list of Weights for the Landmarks in BRAINSConstellationDetector
    """
    input_spec = landmarksConstellationWeightsInputSpec
    output_spec = landmarksConstellationWeightsOutputSpec
    _cmd = ' landmarksConstellationWeights '
    _outputs_filenames = {'outputWeightsList': 'outputWeightsList.wts'}
    _redirect_x = False