import os
from ...base import (
class BRAINSInitializedControlPoints(SEMLikeCommandLine):
    """title: Initialized Control Points (BRAINS)

    category: Utilities.BRAINS

    description: Outputs bspline control points as landmarks

    version: 0.1.0.$Revision: 916 $(alpha)

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Mark Scully

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.  Additional support for Mark Scully and Hans Johnson at the University of Iowa.
    """
    input_spec = BRAINSInitializedControlPointsInputSpec
    output_spec = BRAINSInitializedControlPointsOutputSpec
    _cmd = ' BRAINSInitializedControlPoints '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False