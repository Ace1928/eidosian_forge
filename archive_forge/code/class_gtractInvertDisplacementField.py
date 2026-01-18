import os
from ...base import (
class gtractInvertDisplacementField(SEMLikeCommandLine):
    """title: Invert Displacement Field

    category: Diffusion.GTRACT

    description: This program will invert a deformatrion field. The size of the deformation field is defined by an example image provided by the user

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractInvertDisplacementFieldInputSpec
    output_spec = gtractInvertDisplacementFieldOutputSpec
    _cmd = ' gtractInvertDisplacementField '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd'}
    _redirect_x = False