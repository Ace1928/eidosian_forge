from ..base import TraitedSpec, traits, File
from .base import AFNICommand, AFNICommandInputSpec, AFNICommandOutputSpec
class SVMTrainInputSpec(AFNICommandInputSpec):
    ttype = traits.Str(desc='tname: classification or regression', argstr='-type %s', mandatory=True)
    in_file = File(desc='A 3D+t AFNI brik dataset to be used for training.', argstr='-trainvol %s', mandatory=True, exists=True, copyfile=False)
    out_file = File(name_template='%s_vectors', desc='output sum of weighted linear support vectors file name', argstr='-bucket %s', suffix='_bucket', name_source='in_file')
    model = File(name_template='%s_model', desc='basename for the brik containing the SVM model', argstr='-model %s', suffix='_model', name_source='in_file')
    alphas = File(name_template='%s_alphas', desc='output alphas file name', argstr='-alpha %s', suffix='_alphas', name_source='in_file')
    mask = File(desc='byte-format brik file used to mask voxels in the analysis', argstr='-mask %s', position=-1, exists=True, copyfile=False)
    nomodelmask = traits.Bool(desc='Flag to enable the omission of a mask file', argstr='-nomodelmask')
    trainlabels = File(desc='.1D labels corresponding to the stimulus paradigm for the training data.', argstr='-trainlabels %s', exists=True)
    censor = File(desc='.1D censor file that allows the user to ignore certain samples in the training data.', argstr='-censor %s', exists=True)
    kernel = traits.Str(desc='string specifying type of kernel function:linear, polynomial, rbf, sigmoid', argstr='-kernel %s')
    max_iterations = traits.Int(desc='Specify the maximum number of iterations for the optimization.', argstr='-max_iterations %d')
    w_out = traits.Bool(desc='output sum of weighted linear support vectors', argstr='-wout')
    options = traits.Str(desc='additional options for SVM-light', argstr='%s')