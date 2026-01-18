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
class WatershedSkullStrip(FSCommand):
    """This program strips skull and other outer non-brain tissue and
    produces the brain volume from T1 volume or the scanned volume.

    The "watershed" segmentation algorithm was used to determine the
    intensity values for white matter, grey matter, and CSF.
    A force field was then used to fit a spherical surface to the brain.
    The shape of the surface fit was then evaluated against a previously
    derived template.

    The default parameters are: -w 0.82 -b 0.32 -h 10 -seedpt -ta -wta

    (Segonne 2004)

    Examples
    ========
    >>> from nipype.interfaces.freesurfer import WatershedSkullStrip
    >>> skullstrip = WatershedSkullStrip()
    >>> skullstrip.inputs.in_file = "T1.mgz"
    >>> skullstrip.inputs.t1 = True
    >>> skullstrip.inputs.transform = "transforms/talairach_with_skull.lta"
    >>> skullstrip.inputs.out_file = "brainmask.auto.mgz"
    >>> skullstrip.cmdline
    'mri_watershed -T1 transforms/talairach_with_skull.lta T1.mgz brainmask.auto.mgz'
    """
    _cmd = 'mri_watershed'
    input_spec = WatershedSkullStripInputSpec
    output_spec = WatershedSkullStripOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs