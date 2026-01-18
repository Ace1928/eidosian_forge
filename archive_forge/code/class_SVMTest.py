from ..base import TraitedSpec, traits, File
from .base import AFNICommand, AFNICommandInputSpec, AFNICommandOutputSpec
class SVMTest(AFNICommand):
    """Temporally predictive modeling with the support vector machine
    SVM Test Only
    For complete details, see the `3dsvm Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dsvm.html>`_

    Examples
    ========

    >>> from nipype.interfaces import afni as afni
    >>> svmTest = afni.SVMTest()
    >>> svmTest.inputs.in_file= 'run2+orig'
    >>> svmTest.inputs.model= 'run1+orig_model'
    >>> svmTest.inputs.testlabels= 'run2_categories.1D'
    >>> svmTest.inputs.out_file= 'pred2_model1'
    >>> res = svmTest.run() # doctest: +SKIP

    """
    _cmd = '3dsvm'
    input_spec = SVMTestInputSpec
    output_spec = AFNICommandOutputSpec