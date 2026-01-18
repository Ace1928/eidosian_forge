from nipype.interfaces.base import (
import os
class AffineRegistration(SEMLikeCommandLine):
    """title: Affine Registration

    category: Legacy.Registration

    description: Registers two images together using an affine transform and mutual information. This module is often used to align images of different subjects or images of the same subject from different modalities.

    This module can smooth images prior to registration to mitigate noise and improve convergence. Many of the registration parameters require a working knowledge of the algorithm although the default parameters are sufficient for many registration tasks.



    version: 0.1.0.$Revision: 19608 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/AffineRegistration

    contributor: Daniel Blezek (GE)

    acknowledgements: This module was developed by Daniel Blezek while at GE Research with contributions from Jim Miller.

    This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = AffineRegistrationInputSpec
    output_spec = AffineRegistrationOutputSpec
    _cmd = 'AffineRegistration '
    _outputs_filenames = {'resampledmovingfilename': 'resampledmovingfilename.nii', 'outputtransform': 'outputtransform.txt'}