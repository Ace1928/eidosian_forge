import os
from ...base import (
class BRAINSSnapShotWriter(SEMLikeCommandLine):
    """title: BRAINSSnapShotWriter

    category: Utilities.BRAINS

    description: Create 2D snapshot of input images. Mask images are color-coded

    version: 1.0

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Eunyoung Regina Kim
    """
    input_spec = BRAINSSnapShotWriterInputSpec
    output_spec = BRAINSSnapShotWriterOutputSpec
    _cmd = ' BRAINSSnapShotWriter '
    _outputs_filenames = {'outputFilename': 'outputFilename'}
    _redirect_x = False