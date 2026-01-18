import os
from ...base import (
class BRAINSLandmarkInitializer(SEMLikeCommandLine):
    """title: BRAINSLandmarkInitializer

    category: Utilities.BRAINS

    description: Create transformation file (*mat) from a pair of landmarks (*fcsv) files.

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Eunyoung Regina Kim
    """
    input_spec = BRAINSLandmarkInitializerInputSpec
    output_spec = BRAINSLandmarkInitializerOutputSpec
    _cmd = ' BRAINSLandmarkInitializer '
    _outputs_filenames = {'outputTransformFilename': 'outputTransformFilename'}
    _redirect_x = False