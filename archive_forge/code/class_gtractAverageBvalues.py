import os
from ...base import (
class gtractAverageBvalues(SEMLikeCommandLine):
    """title: Average B-Values

    category: Diffusion.GTRACT

    description: This program will directly average together the baseline gradients (b value equals 0) within a DWI scan. This is usually used after gtractCoregBvalues.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractAverageBvaluesInputSpec
    output_spec = gtractAverageBvaluesOutputSpec
    _cmd = ' gtractAverageBvalues '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd'}
    _redirect_x = False