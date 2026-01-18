from nipype.interfaces.base import (
import os
class ACPCTransform(SEMLikeCommandLine):
    """title: ACPC Transform

    category: Registration.Specialized

    description: <p>Calculate a transformation from two lists of fiducial points.</p><p>ACPC line is two fiducial points, one at the anterior commissure and one at the posterior commissure. The resulting transform will bring the line connecting them to horizontal to the AP axis.</p><p>The midline is a series of points defining the division between the hemispheres of the brain (the mid sagittal plane). The resulting transform will put the output volume with the mid sagittal plane lined up with the AS plane.</p><p>Use the Filtering module<b>Resample Scalar/Vector/DWI Volume</b>to apply the transformation to a volume.</p>

    version: 1.0

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/ACPCTransform

    license: slicer3

    contributor: Nicole Aucoin (SPL, BWH), Ron Kikinis (SPL, BWH)

    acknowledgements: This work is part of the National Alliance for Medical Image Computing (NAMIC), funded by the National Institutes of Health through the NIH Roadmap for Medical Research, Grant U54 EB005149.
    """
    input_spec = ACPCTransformInputSpec
    output_spec = ACPCTransformOutputSpec
    _cmd = 'ACPCTransform '
    _outputs_filenames = {'outputTransform': 'outputTransform.mat'}