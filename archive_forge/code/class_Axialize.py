import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Axialize(AFNICommand):
    """Read in a dataset and write it out as a new dataset
         with the data brick oriented as axial slices.

    For complete details, see the `3dcopy Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3daxialize.html>`__

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> axial3d = afni.Axialize()
    >>> axial3d.inputs.in_file = 'functional.nii'
    >>> axial3d.inputs.out_file = 'axialized.nii'
    >>> axial3d.cmdline
    '3daxialize -prefix axialized.nii functional.nii'
    >>> res = axial3d.run()  # doctest: +SKIP

    """
    _cmd = '3daxialize'
    input_spec = AxializeInputSpec
    output_spec = AFNICommandOutputSpec