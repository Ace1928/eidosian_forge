import os.path as op
from ..base import traits, TraitedSpec, File, InputMultiObject, isdefined
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class FitTensor(MRTrix3Base):
    """
    Convert diffusion-weighted images to tensor images


    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> tsr = mrt.FitTensor()
    >>> tsr.inputs.in_file = 'dwi.mif'
    >>> tsr.inputs.in_mask = 'mask.nii.gz'
    >>> tsr.inputs.grad_fsl = ('bvecs', 'bvals')
    >>> tsr.cmdline                               # doctest: +ELLIPSIS
    'dwi2tensor -fslgrad bvecs bvals -mask mask.nii.gz dwi.mif dti.mif'
    >>> tsr.run()                                 # doctest: +SKIP
    """
    _cmd = 'dwi2tensor'
    input_spec = FitTensorInputSpec
    output_spec = FitTensorOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        if isdefined(self.inputs.predicted_signal):
            outputs['predicted_signal'] = op.abspath(self.inputs.predicted_signal)
        return outputs