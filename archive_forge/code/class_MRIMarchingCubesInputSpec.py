import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIMarchingCubesInputSpec(FSTraitedSpec):
    """
    Uses Freesurfer's mri_mc to create surfaces by tessellating a given input volume
    """
    in_file = File(exists=True, mandatory=True, position=1, argstr='%s', desc='Input volume to tessellate voxels from.')
    label_value = traits.Int(position=2, argstr='%d', mandatory=True, desc='Label value which to tessellate from the input volume. (integer, if input is "filled.mgz" volume, 127 is rh, 255 is lh)')
    connectivity_value = traits.Int(1, position=-1, argstr='%d', usedefault=True, desc='Alter the marching cubes connectivity: 1=6+,2=18,3=6,4=26 (default=1)')
    out_file = File(argstr='./%s', position=-2, genfile=True, desc='output filename or True to generate one')