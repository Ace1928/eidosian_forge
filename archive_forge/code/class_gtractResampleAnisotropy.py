import os
from ...base import (
class gtractResampleAnisotropy(SEMLikeCommandLine):
    """title: Resample Anisotropy

    category: Diffusion.GTRACT

    description: This program will resample a floating point image using either the Rigid or B-Spline transform. You may want to save the aligned B0 image after each of the anisotropy map co-registration steps with the anatomical image to check the registration quality with another tool.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractResampleAnisotropyInputSpec
    output_spec = gtractResampleAnisotropyOutputSpec
    _cmd = ' gtractResampleAnisotropy '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd'}
    _redirect_x = False