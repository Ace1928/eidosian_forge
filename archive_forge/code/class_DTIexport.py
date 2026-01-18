from nipype.interfaces.base import (
import os
class DTIexport(SEMLikeCommandLine):
    """title: DTIexport

    category: Diffusion.Diffusion Data Conversion

    description: Export DTI data to various file formats

    version: 1.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/DTIExport

    contributor: Sonia Pujol (SPL, BWH)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NA-MIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = DTIexportInputSpec
    output_spec = DTIexportOutputSpec
    _cmd = 'DTIexport '
    _outputs_filenames = {'outputFile': 'outputFile'}