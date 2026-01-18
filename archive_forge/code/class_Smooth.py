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
class Smooth(FSCommand):
    """Use FreeSurfer mris_volsmooth to smooth a volume

    This function smoothes cortical regions on a surface and non-cortical
    regions in volume.

    .. note::
       Cortical voxels are mapped to the surface (3D->2D) and then the
       smoothed values from the surface are put back into the volume to fill
       the cortical ribbon. If data is smoothed with this algorithm, one has to
       be careful about how further processing is interpreted.

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import Smooth
    >>> smoothvol = Smooth(in_file='functional.nii', smoothed_file = 'foo_out.nii', reg_file='register.dat', surface_fwhm=10, vol_fwhm=6)
    >>> smoothvol.cmdline
    'mris_volsmooth --i functional.nii --reg register.dat --o foo_out.nii --fwhm 10.000000 --vol-fwhm 6.000000'

    """
    _cmd = 'mris_volsmooth'
    input_spec = SmoothInputSpec
    output_spec = SmoothOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outfile = self.inputs.smoothed_file
        if not isdefined(outfile):
            outfile = self._gen_fname(self.inputs.in_file, suffix='_smooth')
        outputs['smoothed_file'] = outfile
        return outputs

    def _gen_filename(self, name):
        if name == 'smoothed_file':
            return self._list_outputs()[name]
        return None