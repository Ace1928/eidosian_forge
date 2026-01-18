import os
from ...base import (
class gtractConcatDwi(SEMLikeCommandLine):
    """title: Concat DWI Images

    category: Diffusion.GTRACT

    description: This program will concatenate two DTI runs together.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractConcatDwiInputSpec
    output_spec = gtractConcatDwiOutputSpec
    _cmd = ' gtractConcatDwi '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd'}
    _redirect_x = False