import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
class EditWMwithAseg(FSCommand):
    """
    Edits a wm file using a segmentation

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import EditWMwithAseg
    >>> editwm = EditWMwithAseg()
    >>> editwm.inputs.in_file = "T1.mgz"
    >>> editwm.inputs.brain_file = "norm.mgz"
    >>> editwm.inputs.seg_file = "aseg.mgz"
    >>> editwm.inputs.out_file = "wm.asegedit.mgz"
    >>> editwm.inputs.keep_in = True
    >>> editwm.cmdline
    'mri_edit_wm_with_aseg -keep-in T1.mgz norm.mgz aseg.mgz wm.asegedit.mgz'
    """
    _cmd = 'mri_edit_wm_with_aseg'
    input_spec = EditWMwithAsegInputSpec
    output_spec = EditWMwithAsegOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs