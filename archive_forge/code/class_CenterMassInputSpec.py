import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class CenterMassInputSpec(CommandLineInputSpec):
    in_file = File(desc='input file to 3dCM', argstr='%s', position=-2, mandatory=True, exists=True, copyfile=True)
    cm_file = File(name_source='in_file', name_template='%s_cm.out', hash_files=False, keep_extension=False, desc='File to write center of mass to', argstr='> %s', position=-1)
    mask_file = File(desc='Only voxels with nonzero values in the provided mask will be averaged.', argstr='-mask %s', exists=True)
    automask = traits.Bool(desc='Generate the mask automatically', argstr='-automask')
    set_cm = traits.Tuple((traits.Float(), traits.Float(), traits.Float()), desc='After computing the center of mass, set the origin fields in the header so that the center of mass will be at (x,y,z) in DICOM coords.', argstr='-set %f %f %f')
    local_ijk = traits.Bool(desc='Output values as (i,j,k) in local orientation', argstr='-local_ijk')
    roi_vals = traits.List(traits.Int, desc='Compute center of mass for each blob with voxel value of v0, v1, v2, etc. This option is handy for getting ROI centers of mass.', argstr='-roi_vals %s')
    all_rois = traits.Bool(desc="Don't bother listing the values of ROIs you want: The program will find all of them and produce a full list", argstr='-all_rois')