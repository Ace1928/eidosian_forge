import os
from ...base import (
class gtractResampleCodeImage(SEMLikeCommandLine):
    """title: Resample Code Image

    category: Diffusion.GTRACT

    description: This program will resample a short integer code image using either the Rigid or Inverse-B-Spline transform.  The reference image is the DTI tensor anisotropy image space, and the input code image is in anatomical space.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractResampleCodeImageInputSpec
    output_spec = gtractResampleCodeImageOutputSpec
    _cmd = ' gtractResampleCodeImage '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd'}
    _redirect_x = False