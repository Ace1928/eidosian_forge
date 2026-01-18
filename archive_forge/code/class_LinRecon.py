import os
from ...utils.filemanip import split_filename
from ..base import (
class LinRecon(StdOutCommandLine):
    """
    Runs a linear transformation in each voxel.

    Reads  a  linear  transformation from the matrix file assuming the
    imaging scheme specified in the scheme file. Performs the linear
    transformation on the data in every voxel and outputs the result to
    the standard output. The output in every voxel is actually: ::

        [exit code, ln(S(0)), p1, ..., pR]

    where p1, ..., pR are the parameters of the reconstruction.
    Possible exit codes are:

        - 0. No problems.
        - 6. Bad data replaced by substitution of zero.

    The matrix must be R by N+M where N+M is the number of measurements
    and R is the number of parameters of the reconstruction. The matrix
    file contains binary double-precision floats. The matrix elements
    are stored row by row.

    Example
    -------
    First run QBallMX and create a linear transform matrix using
    Spherical Harmonics (sh).

    >>> import nipype.interfaces.camino as cam
    >>> qballmx = cam.QBallMX()
    >>> qballmx.inputs.scheme_file = 'A.scheme'
    >>> qballmx.inputs.basistype = 'sh'
    >>> qballmx.inputs.order = 4
    >>> qballmx.run()            # doctest: +SKIP

    Then run it over each voxel using LinRecon

    >>> qballcoeffs = cam.LinRecon()
    >>> qballcoeffs.inputs.in_file = 'SubjectA.Bfloat'
    >>> qballcoeffs.inputs.scheme_file = 'A.scheme'
    >>> qballcoeffs.inputs.qball_mat = 'A_qmat.Bdouble'
    >>> qballcoeffs.inputs.normalize = True
    >>> qballcoeffs.run()         # doctest: +SKIP

    """
    _cmd = 'linrecon'
    input_spec = LinReconInputSpec
    output_spec = LinReconOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['recon_data'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.scheme_file)
        return name + '_recondata.Bdouble'