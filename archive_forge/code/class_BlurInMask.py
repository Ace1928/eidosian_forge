import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class BlurInMask(AFNICommand):
    """Blurs a dataset spatially inside a mask.  That's all.  Experimental.

    For complete details, see the `3dBlurInMask Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dBlurInMask.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> bim = afni.BlurInMask()
    >>> bim.inputs.in_file = 'functional.nii'
    >>> bim.inputs.mask = 'mask.nii'
    >>> bim.inputs.fwhm = 5.0
    >>> bim.cmdline  # doctest: +ELLIPSIS
    '3dBlurInMask -input functional.nii -FWHM 5.000000 -mask mask.nii -prefix functional_blur'
    >>> res = bim.run()  # doctest: +SKIP

    """
    _cmd = '3dBlurInMask'
    input_spec = BlurInMaskInputSpec
    output_spec = AFNICommandOutputSpec