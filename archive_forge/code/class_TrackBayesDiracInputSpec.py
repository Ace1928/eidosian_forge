import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackBayesDiracInputSpec(TrackInputSpec):
    scheme_file = File(argstr='-schemefile %s', mandatory=True, exists=True, desc='The scheme file corresponding to the data being processed.')
    iterations = traits.Int(argstr='-iterations %d', units='NA', desc='Number of streamlines to generate at each seed point. The default is 5000.')
    pdf = traits.Enum('bingham', 'watson', 'acg', argstr='-pdf %s', desc="Specifies the model for PICo priors (not the curvature priors). The default is 'bingham'.")
    pointset = traits.Int(argstr='-pointset %s', desc='Index to the point set to use for Bayesian likelihood calculation. The index\nspecifies a set of evenly distributed points on the unit sphere, where each point x\ndefines two possible step directions (x or -x) for the streamline path. A larger\nnumber indexes a larger point set, which gives higher angular resolution at the\nexpense of computation time. The default is index 1, which gives 1922 points, index 0\ngives 1082 points, index 2 gives 3002 points.')
    datamodel = traits.Enum('cylsymmdt', 'ballstick', argstr='-datamodel %s', desc='Model of the data for Bayesian tracking. The default model is "cylsymmdt", a diffusion\ntensor with cylindrical symmetry about e_1, ie L1 >= L_2 = L_3. The other model is\n"ballstick", the partial volume model (see ballstickfit).')
    curvepriork = traits.Float(argstr='-curvepriork %G', desc='Concentration parameter for the prior distribution on fibre orientations given the fibre\norientation at the previous step. Larger values of k make curvature less likely.')
    curvepriorg = traits.Float(argstr='-curvepriorg %G', desc='Concentration parameter for the prior distribution on fibre orientations given\nthe fibre orientation at the previous step. Larger values of g make curvature less likely.')
    extpriorfile = File(exists=True, argstr='-extpriorfile %s', desc='Path to a PICo image produced by picopdfs. The PDF in each voxel is used as a prior for\nthe fibre orientation in Bayesian tracking. The prior image must be in the same space\nas the diffusion data.')
    extpriordatatype = traits.Enum('float', 'double', argstr='-extpriordatatype %s', desc='Datatype of the prior image. The default is "double".')