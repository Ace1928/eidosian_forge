import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class NwarpApply(AFNICommandBase):
    """Program to apply a nonlinear 3D warp saved from 3dQwarp
    (or 3dNwarpCat, etc.) to a 3D dataset, to produce a warped
    version of the source dataset.

    For complete details, see the `3dNwarpApply Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNwarpApply.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> nwarp = afni.NwarpApply()
    >>> nwarp.inputs.in_file = 'Fred+orig'
    >>> nwarp.inputs.master = 'NWARP'
    >>> nwarp.inputs.warp = "'Fred_WARP+tlrc Fred.Xaff12.1D'"
    >>> nwarp.cmdline
    "3dNwarpApply -source Fred+orig -interp wsinc5 -master NWARP -prefix Fred+orig_Nwarp -nwarp 'Fred_WARP+tlrc Fred.Xaff12.1D'"
    >>> res = nwarp.run()  # doctest: +SKIP

    """
    _cmd = '3dNwarpApply'
    input_spec = NwarpApplyInputSpec
    output_spec = AFNICommandOutputSpec