import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class To3D(AFNICommand):
    """Create a 3D dataset from 2D image files using AFNI to3d command

    For complete details, see the `to3d Documentation
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/to3d.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> to3d = afni.To3D()
    >>> to3d.inputs.datatype = 'float'
    >>> to3d.inputs.in_folder = '.'
    >>> to3d.inputs.out_file = 'dicomdir.nii'
    >>> to3d.inputs.filetype = 'anat'
    >>> to3d.cmdline  # doctest: +ELLIPSIS
    'to3d -datum float -anat -prefix dicomdir.nii ./*.dcm'
    >>> res = to3d.run()  # doctest: +SKIP

    """
    _cmd = 'to3d'
    input_spec = To3DInputSpec
    output_spec = AFNICommandOutputSpec