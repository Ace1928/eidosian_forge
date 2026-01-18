import os
from ...base import (
class gtractResampleDWIInPlace(SEMLikeCommandLine):
    """title: Resample DWI In Place

    category: Diffusion.GTRACT

    description: Resamples DWI image to structural image.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta, Greg Harris, Hans Johnson, and Joy Matsui.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractResampleDWIInPlaceInputSpec
    output_spec = gtractResampleDWIInPlaceOutputSpec
    _cmd = ' gtractResampleDWIInPlace '
    _outputs_filenames = {'outputResampledB0': 'outputResampledB0.nii', 'outputVolume': 'outputVolume.nii'}
    _redirect_x = False