from nipype.interfaces.base import (
import os
class MultiResolutionAffineRegistration(SEMLikeCommandLine):
    """title: Robust Multiresolution Affine Registration

    category: Legacy.Registration

    description: Provides affine registration using multiple resolution levels and decomposed affine transforms.

    version: 0.1.0.$Revision: 2104 $(alpha)

    documentation-url: http://www.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/MultiResolutionAffineRegistration

    contributor: Casey B Goodlett (Utah)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = MultiResolutionAffineRegistrationInputSpec
    output_spec = MultiResolutionAffineRegistrationOutputSpec
    _cmd = 'MultiResolutionAffineRegistration '
    _outputs_filenames = {'resampledImage': 'resampledImage.nii', 'saveTransform': 'saveTransform.txt'}