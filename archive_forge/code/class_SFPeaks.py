import os
from ...utils.filemanip import split_filename
from ..base import (
class SFPeaks(StdOutCommandLine):
    """
    Finds the peaks of spherical functions.

    This utility reads coefficients of the spherical functions and
    outputs a list of peak directions of the function. It computes the
    value of the function at each of a set of sample points. Then it
    finds local maxima by finding all points at which the function is
    larger than for any other point within a fixed search radius (the
    default  is 0.4). The utility then uses Powell's algorithm to
    optimize the position of each local maximum. Finally the utility
    removes duplicates and tiny peaks with function value smaller than
    some threshold, which is the mean of the function plus some number
    of standard deviations. By default the program checks for con-
    sistency with a second set of starting points, but skips the
    optimization step. To speed up execution, you can turn off the con-
    sistency check by setting the noconsistencycheck flag to True.

    By  default, the utility constructs a set of sample points by
    randomly rotating a unit icosahedron repeatedly (the default is 1000
    times, which produces a set of 6000 points) and concatenating the
    lists of vertices. The 'pointset = <index>' attribute can tell the
    utility to use an evenly distributed set of points (index 0 gives
    1082 points, 1 gives 1922, 2 gives 4322, 3 gives 8672, 4 gives 15872,
    5 gives 32762, 6 gives 72032), which is quicker, because you can get
    away with fewer points. We estimate that you can use a factor of 2.5
    less evenly distributed points than randomly distributed points and
    still expect similar performance levels.

    The output for each voxel is:

    - exitcode (inherited from the input data).
    - ln(A(0))
    - number of peaks found.
    - flag for consistency with a repeated run (number of directions is
      the same and the directions are the same to within a threshold.)
    - mean(f).
    - std(f).
    - direction 1 (x, y, z, f, H00, H01, H10, H11).
    - direction 2 (x, y, z, f, H00, H01, H10, H11).
    - direction 3 (x, y, z, f, H00, H01, H10, H11).

    H is the Hessian of f at the peak. It is the matrix: ::

        [d^2f/ds^2 d^2f/dsdt]
        [d^2f/dtds d^2f/dt^2]
        = [H00 H01]
          [H10 H11]

    where s and t are orthogonal coordinates local to the peak.

    By default the maximum number of peak directions output in each
    voxel is three. If less than three directions are found, zeros are
    output for later directions. The peaks are ordered by the value of
    the function at the peak. If more than the maximum number of
    directions are found only the strongest ones are output. The maximum
    number can be changed setting the 'numpds' attribute.

    The utility can read various kinds of spherical function, but must
    be told what kind of function is input using the 'inputmodel'
    attribute. The description of the 'inputmodel' attribute lists
    additional information required by SFPeaks for each input model.


    Example
    -------
    First run QBallMX and create a linear transform matrix using
    Spherical Harmonics (sh).

    >>> import nipype.interfaces.camino as cam
    >>> sf_peaks = cam.SFPeaks()
    >>> sf_peaks.inputs.in_file = 'A_recon_params.Bdouble'
    >>> sf_peaks.inputs.inputmodel = 'sh'
    >>> sf_peaks.inputs.order = 4
    >>> sf_peaks.inputs.density = 100
    >>> sf_peaks.inputs.searchradius = 1.0
    >>> sf_peaks.run()          # doctest: +SKIP

    """
    _cmd = 'sfpeaks'
    input_spec = SFPeaksInputSpec
    output_spec = SFPeaksOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['peaks'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_peaks.Bdouble'