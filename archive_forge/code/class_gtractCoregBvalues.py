import os
from ...base import (
class gtractCoregBvalues(SEMLikeCommandLine):
    """title: Coregister B-Values

    category: Diffusion.GTRACT

    description: This step should be performed after converting DWI scans from DICOM to NRRD format. This program will register all gradients in a NRRD diffusion weighted 4D vector image (moving image) to a specified index in a fixed image. It also supports co-registration with a T2 weighted image or field map in the same plane as the DWI data. The fixed image for the registration should be a b0 image. A mutual information metric cost function is used for the registration because of the differences in signal intensity as a result of the diffusion gradients. The full affine allows the registration procedure to correct for eddy current distortions that may exist in the data. If the eddyCurrentCorrection is enabled, relaxationFactor (0.25) and maximumStepSize (0.1) should be adjusted.

    version: 4.0.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Modules:GTRACT

    license: http://mri.radiology.uiowa.edu/copyright/GTRACT-Copyright.txt

    contributor: This tool was developed by Vincent Magnotta and Greg Harris.

    acknowledgements: Funding for this version of the GTRACT program was provided by NIH/NINDS R01NS050568-01A2S1
    """
    input_spec = gtractCoregBvaluesInputSpec
    output_spec = gtractCoregBvaluesOutputSpec
    _cmd = ' gtractCoregBvalues '
    _outputs_filenames = {'outputVolume': 'outputVolume.nrrd', 'outputTransform': 'outputTransform.h5'}
    _redirect_x = False