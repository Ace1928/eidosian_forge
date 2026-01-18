import os
from ...base import (
class BRAINSCut(SEMLikeCommandLine):
    """title: BRAINSCut (BRAINS)

    category: Segmentation.Specialized

    description: Automatic Segmentation using neural networks

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Vince Magnotta, Hans Johnson, Greg Harris, Kent Williams, Eunyoung Regina Kim
    """
    input_spec = BRAINSCutInputSpec
    output_spec = BRAINSCutOutputSpec
    _cmd = ' BRAINSCut '
    _outputs_filenames = {}
    _redirect_x = False