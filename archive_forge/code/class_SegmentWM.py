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
class SegmentWM(FSCommand):
    """
    This program segments white matter from the input volume.  The input
    volume should be normalized such that white matter voxels are
    ~110-valued, and the volume is conformed to 256^3.


    Examples
    ========
    >>> from nipype.interfaces import freesurfer
    >>> SegmentWM_node = freesurfer.SegmentWM()
    >>> SegmentWM_node.inputs.in_file = "norm.mgz"
    >>> SegmentWM_node.inputs.out_file = "wm.seg.mgz"
    >>> SegmentWM_node.cmdline
    'mri_segment norm.mgz wm.seg.mgz'
    """
    _cmd = 'mri_segment'
    input_spec = SegmentWMInputSpec
    output_spec = SegmentWMOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs