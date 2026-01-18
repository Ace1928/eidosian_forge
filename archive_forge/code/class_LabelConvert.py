import os
import os.path as op
from ..base import CommandLineInputSpec, traits, TraitedSpec, File, isdefined
from .base import MRTrix3Base
class LabelConvert(MRTrix3Base):
    """
    Re-configure parcellation to be incrementally defined.

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> labels = mrt.LabelConvert()
    >>> labels.inputs.in_file = 'aparc+aseg.nii'
    >>> labels.inputs.in_config = 'mrtrix3_labelconfig.txt'
    >>> labels.inputs.in_lut = 'FreeSurferColorLUT.txt'
    >>> labels.cmdline
    'labelconvert aparc+aseg.nii FreeSurferColorLUT.txt mrtrix3_labelconfig.txt parcellation.mif'
    >>> labels.run()                                 # doctest: +SKIP
    """
    _cmd = 'labelconvert'
    input_spec = LabelConvertInputSpec
    output_spec = LabelConvertOutputSpec

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if not isdefined(self.inputs.in_config):
            from nipype.utils.filemanip import which
            path = which(self._cmd)
            if path is None:
                path = os.getenv(MRTRIX3_HOME, '/opt/mrtrix3')
            else:
                path = op.dirname(op.dirname(path))
            self.inputs.in_config = op.join(path, 'src/dwi/tractography/connectomics/example_configs/fs_default.txt')
        return super(LabelConvert, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs