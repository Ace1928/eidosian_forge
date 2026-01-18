import os.path as op
from ..base import (
from .base import MRTrix3Base, MRTrix3BaseInputSpec
class DWIBiasCorrect(MRTrix3Base):
    """
    Perform B1 field inhomogeneity correction for a DWI volume series.

    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/scripts/dwibiascorrect.html>

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> bias_correct = mrt.DWIBiasCorrect()
    >>> bias_correct.inputs.in_file = 'dwi.mif'
    >>> bias_correct.inputs.use_ants = True
    >>> bias_correct.cmdline
    'dwibiascorrect ants dwi.mif dwi_biascorr.mif'
    >>> bias_correct.run()                             # doctest: +SKIP
    """
    _cmd = 'dwibiascorrect'
    input_spec = DWIBiasCorrectInputSpec
    output_spec = DWIBiasCorrectOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name in ('use_ants', 'use_fsl'):
            ver = self.version
            if ver is not None and (ver[0] < '3' or ver.startswith('3.0_RC')):
                return f'-{trait_spec.argstr}'
        return super()._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.out_file:
            outputs['out_file'] = op.abspath(self.inputs.out_file)
        if self.inputs.bias:
            outputs['bias'] = op.abspath(self.inputs.bias)
        return outputs