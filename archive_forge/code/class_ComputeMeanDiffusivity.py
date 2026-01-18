import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeMeanDiffusivity(StdOutCommandLine):
    """
    Computes the mean diffusivity (trace/3) from diffusion tensors.

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> md = cmon.ComputeMeanDiffusivity()
    >>> md.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> md.inputs.scheme_file = 'A.scheme'
    >>> md.run()                  # doctest: +SKIP

    """
    _cmd = 'md'
    input_spec = ComputeMeanDiffusivityInputSpec
    output_spec = ComputeMeanDiffusivityOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['md'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_MD.img'