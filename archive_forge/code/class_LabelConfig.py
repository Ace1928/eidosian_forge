import os
import os.path as op
from ..base import CommandLineInputSpec, traits, TraitedSpec, File, isdefined
from .base import MRTrix3Base
class LabelConfig(MRTrix3Base):
    """
    Re-configure parcellation to be incrementally defined.

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> labels = mrt.LabelConfig()
    >>> labels.inputs.in_file = 'aparc+aseg.nii'
    >>> labels.inputs.in_config = 'mrtrix3_labelconfig.txt'
    >>> labels.cmdline                               # doctest: +ELLIPSIS
    'labelconfig aparc+aseg.nii mrtrix3_labelconfig.txt parcellation.mif'
    >>> labels.run()                                 # doctest: +SKIP
    """
    _cmd = 'labelconfig'
    input_spec = LabelConfigInputSpec
    output_spec = LabelConfigOutputSpec

    def _parse_inputs(self, skip=None):
        if skip is None:
            skip = []
        if not isdefined(self.inputs.in_config):
            from shutil import which
            path = which(self._cmd)
            if path is None:
                path = os.getenv(MRTRIX3_HOME, '/opt/mrtrix3')
            else:
                path = op.dirname(op.dirname(path))
            self.inputs.in_config = op.join(path, 'src/dwi/tractography/connectomics/example_configs/fs_default.txt')
        return super(LabelConfig, self)._parse_inputs(skip=skip)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs