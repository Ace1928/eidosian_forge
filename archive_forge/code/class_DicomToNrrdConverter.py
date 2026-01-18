from nipype.interfaces.base import (
import os
class DicomToNrrdConverter(SEMLikeCommandLine):
    """title: DICOM to NRRD Converter

    category: Converters

    description: Converts diffusion weighted MR images in dicom series into Nrrd format for analysis in Slicer. This program has been tested on only a limited subset of DTI dicom formats available from Siemens, GE, and Phillips scanners. Work in progress to support dicom multi-frame data. The program parses dicom header to extract necessary information about measurement frame, diffusion weighting directions, b-values, etc, and write out a nrrd image. For non-diffusion weighted dicom images, it loads in an entire dicom series and writes out a single dicom volume in a .nhdr/.raw pair.

    version: 0.2.0.$Revision: 916 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/DicomToNrrdConverter

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Xiaodong Tao (GE), Vince Magnotta (UIowa), Hans Johnson (UIowa)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.  Additional support for DTI data produced on Philips scanners was contributed by Vincent Magnotta and Hans Johnson at the University of Iowa.
    """
    input_spec = DicomToNrrdConverterInputSpec
    output_spec = DicomToNrrdConverterOutputSpec
    _cmd = 'DicomToNrrdConverter '
    _outputs_filenames = {'outputDirectory': 'outputDirectory'}