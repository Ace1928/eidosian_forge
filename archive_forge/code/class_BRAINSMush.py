import os
from ...base import (
class BRAINSMush(SEMLikeCommandLine):
    """title: Brain Extraction from T1/T2 image (BRAINS)

    category: Utilities.BRAINS

    description: This program: 1) generates a weighted mixture image optimizing the mean and variance and 2) produces a mask of the brain volume

    version: 0.1.0.$Revision: 1.4 $(alpha)

    documentation-url: http:://mri.radiology.uiowa.edu

    license: https://www.nitrc.org/svn/brains/BuildScripts/trunk/License.txt

    contributor: This tool is a modification by Steven Dunn of a program developed by Greg Harris and Ron Pierson.

    acknowledgements: This work was developed by the University of Iowa Departments of Radiology and Psychiatry. This software was supported in part of NIH/NINDS award NS050568.
    """
    input_spec = BRAINSMushInputSpec
    output_spec = BRAINSMushOutputSpec
    _cmd = ' BRAINSMush '
    _outputs_filenames = {'outputMask': 'outputMask.nii.gz', 'outputWeightsFile': 'outputWeightsFile.txt', 'outputVolume': 'outputVolume.nii.gz'}
    _redirect_x = False