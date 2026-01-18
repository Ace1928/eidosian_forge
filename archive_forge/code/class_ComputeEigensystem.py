import os
from ...utils.filemanip import split_filename
from ..base import (
class ComputeEigensystem(StdOutCommandLine):
    """
    Computes the eigensystem from tensor fitted data.

    Reads diffusion tensor (single, two-tensor, three-tensor or multitensor) data from the
    standard input, computes the eigenvalues and eigenvectors of each tensor and outputs the
    results to the standard output. For multiple-tensor data the program outputs the
    eigensystem of each tensor. For each tensor the program outputs: {l_1, e_11, e_12, e_13,
    l_2, e_21, e_22, e_33, l_3, e_31, e_32, e_33}, where l_1 >= l_2 >= l_3 and e_i = (e_i1,
    e_i2, e_i3) is the eigenvector with eigenvalue l_i. For three-tensor data, for example,
    the output contains thirty-six values per voxel.

    Example
    -------

    >>> import nipype.interfaces.camino as cmon
    >>> dteig = cmon.ComputeEigensystem()
    >>> dteig.inputs.in_file = 'tensor_fitted_data.Bdouble'
    >>> dteig.run()                  # doctest: +SKIP
    """
    _cmd = 'dteig'
    input_spec = ComputeEigensystemInputSpec
    output_spec = ComputeEigensystemOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['eigen'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        datatype = self.inputs.outputdatatype
        return name + '_eig.B' + datatype