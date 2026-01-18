import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpPointsFromStdInputSpec(CommandLineInputSpec):
    img_file = File(exists=True, argstr='-img %s', mandatory=True, desc='filename of a destination image')
    std_file = File(exists=True, argstr='-std %s', mandatory=True, desc='filename of the image in standard space')
    in_coords = File(exists=True, position=-2, argstr='%s', mandatory=True, desc='filename of file containing coordinates')
    xfm_file = File(exists=True, argstr='-xfm %s', xor=['warp_file'], desc='filename of affine transform (e.g. source2dest.mat)')
    warp_file = File(exists=True, argstr='-warp %s', xor=['xfm_file'], desc='filename of warpfield (e.g. intermediate2dest_warp.nii.gz)')
    coord_vox = traits.Bool(True, argstr='-vox', xor=['coord_mm'], desc='all coordinates in voxels - default')
    coord_mm = traits.Bool(False, argstr='-mm', xor=['coord_vox'], desc='all coordinates in mm')