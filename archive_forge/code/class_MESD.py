import os
from ...utils.filemanip import split_filename
from ..base import (
class MESD(StdOutCommandLine):
    """
    MESD is a general program for maximum entropy spherical deconvolution.
    It also runs PASMRI, which is a special case of spherical deconvolution.
    The input data must be in voxel order.

    The format of the output in each voxel is:
    { exitcode, ln(A^star(0)), lambda_0, lambda_1, ..., lambda_N }

    The  exitcode  contains  the  results of three tests. The first test thresholds
    the maximum relative error between the numerical integrals computed at con-
    vergence and those computed using a larger test point set; if the error is
    greater than a threshold the exitcode is increased from zero to one as a
    warning; if it is greater than a larger threshold the exitcode is increased to
    two to suggest failure. The  second  test  thresholds  the predicted  error in
    numerical integrals computed using the test point set; if the predicted error
    is greater than a threshold the exitcode is increased by 10. The third test
    thresholds the RMS error between the measurements and their predictions from
    the fitted deconvolution; if the errors are greater than a threshold, the exit
    code is increased by 100. An exitcode of 112 means that all three tests were
    failed and the result is likely to be unreliable.  If all is well the exitcode
    is zero. Results are often still reliable even if one or two of the tests are
    failed.

    Other possible exitcodes are:

        - 5 - The optimization failed to converge
        - -1 - Background
        - -100 - Something wrong in the MRI data, e.g. negative or zero measurements,
          so that the optimization could not run.

    The  standard  MESD  implementation  is computationally demanding, particularly
    as the number of measurements increases (computation is approximately O(N^2),
    where N is the number of measurements). There are two ways to obtain significant
    computational speed-up:

    i) Turn off error checks and use a small point set for computing numerical
    integrals in the algorithm by adding the flag -fastmesd. Sakaie CDMRI 2008
    shows that using the smallest point set  (-basepointset  0)  with  no
    error  checks  usually  has only a minor effect on the output of the algorithm,
    but provides a major reduction in computation time. You can increase the point
    set size using -basepointset with an argument higher than 0, which may produce
    better results in some voxels, but will increase computation time, which
    approximately doubles every time the point set index increases by 1.

    ii) Reduce the complexity of the maximum entropy encoding using -mepointset <X>.
    By default <X> = N, the number of measurements, and is the number of parameters
    in the max.  ent. representation of the  output  function, ie  the  number of
    lambda parameters, as described in Jansons and Alexander Inverse Problems 2003.
    However, we can represent the function using less components and <X> here
    specifies the number of lambda parameters. To obtain speed-up, set <X>
    < N; complexity become O(<X>^2) rather than O(N^2). Note that <X> must be chosen
    so that the camino/PointSets directory contains a point set  with  that  number
    of  elements.  When  -mepointset decreases, the  numerical  integration checks
    make less and less of a difference and smaller point sets for numerical
    integration (see -basepointset) become adequate. So when <X> is low -fastmesd is
    worth using to get even more speed-up.

    The choice of <X> is a parameter of the technique. Too low and you lose angular
    resoloution; too high and you see no computational benefit and may even suffer
    from overfitting. Empirically, we  have  found  that  <X>=16 often  gives  good
    results and good speed up, but it is worth trying a few values a comparing
    performance. The reduced encoding is described in the following ISMRM abstract:
    Sweet and Alexander "Reduced Encoding Persistent Angular Structure" 572 ISMRM 2010.

    Example
    -------
    Run MESD on every voxel of the data file SubjectA.Bfloat using the PASMRI kernel.

    >>> import nipype.interfaces.camino as cam
    >>> mesd = cam.MESD()
    >>> mesd.inputs.in_file = 'SubjectA.Bfloat'
    >>> mesd.inputs.scheme_file = 'A.scheme'
    >>> mesd.inputs.inverter = 'PAS'
    >>> mesd.inputs.inverter_param = 1.4
    >>> mesd.run()            # doctest: +SKIP

    """
    _cmd = 'mesd'
    input_spec = MESDInputSpec
    output_spec = MESDOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['mesd_data'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.scheme_file)
        return name + '_MESD.Bdouble'