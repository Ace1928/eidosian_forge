import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Autobox(AFNICommand):
    """Computes size of a box that fits around the volume.
    Also can be used to crop the volume to that box.

    For complete details, see the `3dAutobox Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAutobox.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> abox = afni.Autobox()
    >>> abox.inputs.in_file = 'structural.nii'
    >>> abox.inputs.padding = 5
    >>> abox.cmdline
    '3dAutobox -input structural.nii -prefix structural_autobox -npad 5'
    >>> res = abox.run()  # doctest: +SKIP

    """
    _cmd = '3dAutobox'
    input_spec = AutoboxInputSpec
    output_spec = AutoboxOutputSpec

    def aggregate_outputs(self, runtime=None, needed_outputs=None):
        outputs = super(Autobox, self).aggregate_outputs(runtime, needed_outputs)
        pattern = 'x=(?P<x_min>-?\\d+)\\.\\.(?P<x_max>-?\\d+)  y=(?P<y_min>-?\\d+)\\.\\.(?P<y_max>-?\\d+)  z=(?P<z_min>-?\\d+)\\.\\.(?P<z_max>-?\\d+)'
        for line in runtime.stderr.split('\n'):
            m = re.search(pattern, line)
            if m:
                d = m.groupdict()
                outputs.trait_set(**{k: int(d[k]) for k in d.keys()})
        return outputs