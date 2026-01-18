import os
from ...utils.filemanip import split_filename
from ..base import (
class DTIFit(StdOutCommandLine):
    """
    Reads diffusion MRI data, acquired using the acquisition scheme detailed in the scheme file,
    from the data file.

    Use non-linear fitting instead of the default linear regression to the log measurements.
    The data file stores the diffusion MRI data in voxel order with the measurements stored
    in big-endian format and ordered as in the scheme file.
    The default input data type is four-byte float.
    The default output data type is eight-byte double.
    See modelfit and camino for the format of the data file and scheme file.
    The program fits the diffusion tensor to each voxel and outputs the results,
    in voxel order and as big-endian eight-byte doubles, to the standard output.
    The program outputs eight values in each voxel:
    [exit code, ln(S(0)), D_xx, D_xy, D_xz, D_yy, D_yz, D_zz].
    An exit code of zero indicates no problems.
    For a list of other exit codes, see modelfit(1).
    The entry S(0) is an estimate of the signal at q=0.

    Example
    -------
    >>> import nipype.interfaces.camino as cmon
    >>> fit = cmon.DTIFit()
    >>> fit.inputs.scheme_file = 'A.scheme'
    >>> fit.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> fit.run()                  # doctest: +SKIP

    """
    _cmd = 'dtfit'
    input_spec = DTIFitInputSpec
    output_spec = DTIFitOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['tensor_fitted'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_DT.Bdouble'