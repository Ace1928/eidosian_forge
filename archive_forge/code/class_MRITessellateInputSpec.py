import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRITessellateInputSpec(FSTraitedSpec):
    """
    Uses Freesurfer's mri_tessellate to create surfaces by tessellating a given input volume
    """
    in_file = File(exists=True, mandatory=True, position=-3, argstr='%s', desc='Input volume to tessellate voxels from.')
    label_value = traits.Int(position=-2, argstr='%d', mandatory=True, desc='Label value which to tessellate from the input volume. (integer, if input is "filled.mgz" volume, 127 is rh, 255 is lh)')
    out_file = File(argstr='%s', position=-1, genfile=True, desc='output filename or True to generate one')
    tesselate_all_voxels = traits.Bool(argstr='-a', desc='Tessellate the surface of all voxels with different labels')
    use_real_RAS_coordinates = traits.Bool(argstr='-n', desc='Saves surface with real RAS coordinates where c_(r,a,s) != 0')