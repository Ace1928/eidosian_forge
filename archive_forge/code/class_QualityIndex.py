import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class QualityIndex(CommandLine):
    """Computes a quality index for each sub-brick in a 3D+time dataset.
    The output is a 1D time series with the index for each sub-brick.
    The results are written to stdout.

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> tqual = afni.QualityIndex()
    >>> tqual.inputs.in_file = 'functional.nii'
    >>> tqual.cmdline  # doctest: +ELLIPSIS
    '3dTqual functional.nii > functional_tqual'
    >>> res = tqual.run()  # doctest: +SKIP

    See Also
    --------
    For complete details, see the `3dTqual Documentation
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTqual.html>`_

    """
    _cmd = '3dTqual'
    input_spec = QualityIndexInputSpec
    output_spec = QualityIndexOutputSpec