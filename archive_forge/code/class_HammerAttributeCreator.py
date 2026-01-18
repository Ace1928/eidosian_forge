import os
from ...base import (
class HammerAttributeCreator(SEMLikeCommandLine):
    """title: HAMMER Feature Vectors

    category: Filtering.FeatureDetection

    description: Create the feature vectors used by HAMMER.

    version: 0.1.0.$Revision: 1 $(alpha)

    documentation-url: http:://www.na-mic.org/

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This was extracted from the Hammer Registration source code, and wrapped up by Hans J. Johnson.
    """
    input_spec = HammerAttributeCreatorInputSpec
    output_spec = HammerAttributeCreatorOutputSpec
    _cmd = ' HammerAttributeCreator '
    _outputs_filenames = {}
    _redirect_x = False