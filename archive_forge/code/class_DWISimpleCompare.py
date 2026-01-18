import os
from ..base import (
class DWISimpleCompare(SEMLikeCommandLine):
    """title: Nrrd DWI comparison

    category: Converters

    description: Compares two nrrd format DWI images and verifies that gradient magnitudes, gradient directions, measurement frame, and max B0 value are identicle.  Used for testing DWIConvert.

    version: 0.1.0.$Revision: 916 $(alpha)

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/DWIConvert

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: Mark Scully (UIowa)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.  Additional support for DTI data produced on Philips scanners was contributed by Vincent Magnotta and Hans Johnson at the University of Iowa.
    """
    input_spec = DWISimpleCompareInputSpec
    output_spec = DWISimpleCompareOutputSpec
    _cmd = ' DWISimpleCompare '
    _outputs_filenames = {}
    _redirect_x = False