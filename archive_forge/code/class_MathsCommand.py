import os
from ..base import (
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
from ...utils.filemanip import split_filename
class MathsCommand(NiftySegCommand):
    """
    Base Command Interface for seg_maths interfaces.

    The executable seg_maths enables the sequential execution of arithmetic
    operations, like multiplication (-mul), division (-div) or addition
    (-add), binarisation (-bin) or thresholding (-thr) operations and
    convolution by a Gaussian kernel (-smo). It also allows mathematical
    morphology based operations like dilation (-dil), erosion (-ero),
    connected components (-lconcomp) and hole filling (-fill), Euclidean
    (- euc) and geodesic (-geo) distance transforms, local image similarity
    metric calculation (-lncc and -lssd). Finally, it allows multiple
    operations over the dimensionality of the image, from merging 3D images
    together as a 4D image (-merge) or splitting (-split or -tp) 4D images
    into several 3D images, to estimating the maximum, minimum and average
    over all time-points, etc.
    """
    _cmd = get_custom_path('seg_maths', env_dir='NIFTYSEGDIR')
    input_spec = MathsInput
    output_spec = MathsOutput
    _suffix = '_maths'

    def _overload_extension(self, value, name=None):
        path, base, _ = split_filename(value)
        _, _, ext = split_filename(self.inputs.in_file)
        suffix = self._suffix
        if suffix != '_merged' and isdefined(self.inputs.operation):
            suffix = '_' + self.inputs.operation
        return os.path.join(path, '{0}{1}{2}'.format(base, suffix, ext))