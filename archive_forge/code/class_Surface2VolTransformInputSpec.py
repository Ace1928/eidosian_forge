import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class Surface2VolTransformInputSpec(FSTraitedSpec):
    source_file = File(exists=True, argstr='--surfval %s', copyfile=False, mandatory=True, xor=['mkmask'], desc='This is the source of the surface values')
    hemi = traits.Str(argstr='--hemi %s', mandatory=True, desc='hemisphere of data')
    transformed_file = File(name_template='%s_asVol.nii', desc='Output volume', argstr='--outvol %s', name_source=['source_file'], hash_files=False)
    reg_file = File(exists=True, argstr='--volreg %s', mandatory=True, desc='tkRAS-to-tkRAS matrix   (tkregister2 format)', xor=['subject_id'])
    template_file = File(exists=True, argstr='--template %s', desc='Output template volume')
    mkmask = traits.Bool(desc='make a mask instead of loading surface values', argstr='--mkmask', xor=['source_file'])
    vertexvol_file = File(name_template='%s_asVol_vertex.nii', desc='Path name of the vertex output volume, which is the same as output volume except that the value of each voxel is the vertex-id that is mapped to that voxel.', argstr='--vtxvol %s', name_source=['source_file'], hash_files=False)
    surf_name = traits.Str(argstr='--surf %s', desc='surfname (default is white)')
    projfrac = traits.Float(argstr='--projfrac %s', desc='thickness fraction')
    subjects_dir = traits.Str(argstr='--sd %s', desc='freesurfer subjects directory defaults to $SUBJECTS_DIR')
    subject_id = traits.Str(argstr='--identity %s', desc='subject id', xor=['reg_file'])