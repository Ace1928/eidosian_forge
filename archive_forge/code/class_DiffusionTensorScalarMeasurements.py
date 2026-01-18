from nipype.interfaces.base import (
import os
class DiffusionTensorScalarMeasurements(SEMLikeCommandLine):
    """title: Diffusion Tensor Scalar Measurements

    category: Diffusion.Diffusion Tensor Images

    description: Compute a set of different scalar measurements from a tensor field, specially oriented for Diffusion Tensors where some rotationally invariant measurements, like Fractional Anisotropy, are highly used to describe the anistropic behaviour of the tensor.

    version: 0.1.0.$Revision: 1892 $(alpha)

    documentation-url: http://wiki.slicer.org/slicerWiki/index.php/Documentation/4.1/Modules/DiffusionTensorMathematics

    contributor: Raul San Jose (SPL, BWH)

    acknowledgements: LMI
    """
    input_spec = DiffusionTensorScalarMeasurementsInputSpec
    output_spec = DiffusionTensorScalarMeasurementsOutputSpec
    _cmd = 'DiffusionTensorScalarMeasurements '
    _outputs_filenames = {'outputScalar': 'outputScalar.nii'}