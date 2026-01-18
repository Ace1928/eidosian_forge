import os
from ...base import (
class BRAINSTalairachMask(SEMLikeCommandLine):
    """title: Talairach Mask

    category: BRAINS.Segmentation

    description: This program creates a binary image representing the specified Talairach region. The input is an example image to define the physical space for the resulting image, the Talairach grid representation in VTK format, and the file containing the Talairach box definitions to be generated. These can be combined in BRAINS to create a label map using the procedure Brains::WorkupUtils::CreateLabelMapFromBinaryImages.

    version: 0.1

    documentation-url: http://www.nitrc.org/plugins/mwiki/index.php/brains:BRAINSTalairachMask

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Steven Dunn and Vincent Magnotta

    acknowledgements: Funding for this work was provided by NIH/NINDS award NS050568
    """
    input_spec = BRAINSTalairachMaskInputSpec
    output_spec = BRAINSTalairachMaskOutputSpec
    _cmd = ' BRAINSTalairachMask '
    _outputs_filenames = {'outputVolume': 'outputVolume.nii'}
    _redirect_x = False