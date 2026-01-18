from nipype.interfaces.base import (
import os
class DTIimport(SEMLikeCommandLine):
    """title: DTIimport

    category: Diffusion.Diffusion Data Conversion

    description: Import tensor datasets from various formats, including the NifTi file format

    version: 1.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/DTIImport

    contributor: Sonia Pujol (SPL, BWH)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NA-MIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = DTIimportInputSpec
    output_spec = DTIimportOutputSpec
    _cmd = 'DTIimport '
    _outputs_filenames = {'outputTensor': 'outputTensor.nii'}