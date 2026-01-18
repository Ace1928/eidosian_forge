from ..base import TraitedSpec, traits, File
from .base import AFNICommand, AFNICommandInputSpec, AFNICommandOutputSpec
class SVMTrain(AFNICommand):
    """Temporally predictive modeling with the support vector machine
    SVM Train Only
    For complete details, see the `3dsvm Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dsvm.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> svmTrain = afni.SVMTrain()
    >>> svmTrain.inputs.in_file = 'run1+orig'
    >>> svmTrain.inputs.trainlabels = 'run1_categories.1D'
    >>> svmTrain.inputs.ttype = 'regression'
    >>> svmTrain.inputs.mask = 'mask.nii'
    >>> svmTrain.inputs.model = 'model_run1'
    >>> svmTrain.inputs.alphas = 'alphas_run1'
    >>> res = svmTrain.run() # doctest: +SKIP

    """
    _cmd = '3dsvm'
    input_spec = SVMTrainInputSpec
    output_spec = SVMTrainOutputSpec
    _additional_metadata = ['suffix']

    def _format_arg(self, name, trait_spec, value):
        return super(SVMTrain, self)._format_arg(name, trait_spec, value)