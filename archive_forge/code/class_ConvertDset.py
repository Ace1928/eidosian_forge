import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class ConvertDset(AFNICommandBase):
    """Converts a surface dataset from one format to another.

    For complete details, see the `ConvertDset Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/ConvertDset.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> convertdset = afni.ConvertDset()
    >>> convertdset.inputs.in_file = 'lh.pial_converted.gii'
    >>> convertdset.inputs.out_type = 'niml_asc'
    >>> convertdset.inputs.out_file = 'lh.pial_converted.niml.dset'
    >>> convertdset.cmdline
    'ConvertDset -o_niml_asc -input lh.pial_converted.gii -prefix lh.pial_converted.niml.dset'
    >>> res = convertdset.run()  # doctest: +SKIP

    """
    _cmd = 'ConvertDset'
    input_spec = ConvertDsetInputSpec
    output_spec = AFNICommandOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs