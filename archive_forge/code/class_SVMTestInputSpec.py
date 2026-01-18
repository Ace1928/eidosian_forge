from ..base import TraitedSpec, traits, File
from .base import AFNICommand, AFNICommandInputSpec, AFNICommandOutputSpec
class SVMTestInputSpec(AFNICommandInputSpec):
    model = traits.Str(desc='modname is the basename for the brik containing the SVM model', argstr='-model %s', mandatory=True)
    in_file = File(desc='A 3D or 3D+t AFNI brik dataset to be used for testing.', argstr='-testvol %s', exists=True, mandatory=True)
    out_file = File(name_template='%s_predictions', desc='filename for .1D prediction file(s).', argstr='-predictions %s')
    testlabels = File(desc='*true* class category .1D labels for the test dataset. It is used to calculate the prediction accuracy performance', exists=True, argstr='-testlabels %s')
    classout = traits.Bool(desc='Flag to specify that pname files should be integer-valued, corresponding to class category decisions.', argstr='-classout')
    nopredcensord = traits.Bool(desc='Flag to prevent writing predicted values for censored time-points', argstr='-nopredcensord')
    nodetrend = traits.Bool(desc='Flag to specify that pname files should not be linearly detrended', argstr='-nodetrend')
    multiclass = traits.Bool(desc='Specifies multiclass algorithm for classification', argstr='-multiclass %s')
    options = traits.Str(desc='additional options for SVM-light', argstr='%s')