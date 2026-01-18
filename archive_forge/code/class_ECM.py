import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class ECM(AFNICommand):
    """Performs degree centrality on a dataset using a given maskfile
    via the 3dECM command

    For complete details, see the `3dECM Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dECM.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> ecm = afni.ECM()
    >>> ecm.inputs.in_file = 'functional.nii'
    >>> ecm.inputs.mask = 'mask.nii'
    >>> ecm.inputs.sparsity = 0.1 # keep top 0.1% of connections
    >>> ecm.inputs.out_file = 'out.nii'
    >>> ecm.cmdline
    '3dECM -mask mask.nii -prefix out.nii -sparsity 0.100000 functional.nii'
    >>> res = ecm.run()  # doctest: +SKIP

    """
    _cmd = '3dECM'
    input_spec = ECMInputSpec
    output_spec = AFNICommandOutputSpec