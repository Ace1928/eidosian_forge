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
class ConcatenateLTA(FSCommand):
    """Concatenates two consecutive LTA transformations into one overall
    transformation

    Out = LTA2*LTA1

    Examples
    --------
    >>> from nipype.interfaces.freesurfer import ConcatenateLTA
    >>> conc_lta = ConcatenateLTA()
    >>> conc_lta.inputs.in_lta1 = 'lta1.lta'
    >>> conc_lta.inputs.in_lta2 = 'lta2.lta'
    >>> conc_lta.cmdline
    'mri_concatenate_lta lta1.lta lta2.lta lta1_concat.lta'

    You can use 'identity.nofile' as the filename for in_lta2, e.g.:

    >>> conc_lta.inputs.in_lta2 = 'identity.nofile'
    >>> conc_lta.inputs.invert_1 = True
    >>> conc_lta.inputs.out_file = 'inv1.lta'
    >>> conc_lta.cmdline
    'mri_concatenate_lta -invert1 lta1.lta identity.nofile inv1.lta'

    To create a RAS2RAS transform:

    >>> conc_lta.inputs.out_type = 'RAS2RAS'
    >>> conc_lta.cmdline
    'mri_concatenate_lta -invert1 -out_type 1 lta1.lta identity.nofile inv1.lta'
    """
    _cmd = 'mri_concatenate_lta'
    input_spec = ConcatenateLTAInputSpec
    output_spec = ConcatenateLTAOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'out_type':
            value = {'VOX2VOX': 0, 'RAS2RAS': 1}[value]
        return super(ConcatenateLTA, self)._format_arg(name, spec, value)