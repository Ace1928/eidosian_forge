from nipype.interfaces.base import (
import os
class N4ITKBiasFieldCorrection(SEMLikeCommandLine):
    """title: N4ITK MRI Bias correction

    category: Filtering

    description: Performs image bias correction using N4 algorithm. This module is based on the ITK filters contributed in the following publication:  Tustison N, Gee J "N4ITK: Nick's N3 ITK Implementation For MRI Bias Field Correction", The Insight Journal 2009 January-June, http://hdl.handle.net/10380/3053

    version: 9

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/N4ITKBiasFieldCorrection

    contributor: Nick Tustison (UPenn), Andrey Fedorov (SPL, BWH), Ron Kikinis (SPL, BWH)

    acknowledgements: The development of this module was partially supported by NIH grants R01 AA016748-01, R01 CA111288 and U01 CA151261 as well as by NA-MIC, NAC, NCIGT and the Slicer community.
    """
    input_spec = N4ITKBiasFieldCorrectionInputSpec
    output_spec = N4ITKBiasFieldCorrectionOutputSpec
    _cmd = 'N4ITKBiasFieldCorrection '
    _outputs_filenames = {'outputimage': 'outputimage.nii', 'outputbiasfield': 'outputbiasfield.nii'}