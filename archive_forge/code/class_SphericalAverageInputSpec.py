import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class SphericalAverageInputSpec(FSTraitedSpec):
    out_file = File(argstr='%s', genfile=True, exists=False, position=-1, desc='Output filename')
    in_average = Directory(argstr='%s', exists=True, genfile=True, position=-2, desc='Average subject')
    in_surf = File(argstr='%s', mandatory=True, exists=True, position=-3, desc='Input surface file')
    hemisphere = traits.Enum('lh', 'rh', argstr='%s', mandatory=True, position=-4, desc='Input hemisphere')
    fname = traits.String(argstr='%s', mandatory=True, position=-5, desc="Filename from the average subject directory.\nExample: to use rh.entorhinal.label as the input label filename, set fname to 'rh.entorhinal'\nand which to 'label'. The program will then search for\n``<in_average>/label/rh.entorhinal.label``")
    which = traits.Enum('coords', 'label', 'vals', 'curv', 'area', argstr='%s', mandatory=True, position=-6, desc='No documentation')
    subject_id = traits.String(argstr='-o %s', mandatory=True, desc='Output subject id')
    erode = traits.Int(argstr='-erode %d', desc='Undocumented')
    in_orig = File(argstr='-orig %s', exists=True, desc='Original surface filename')
    threshold = traits.Float(argstr='-t %.1f', desc='Undocumented')