import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class EMRegisterInputSpec(FSTraitedSpecOpenMP):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=-3, desc='in brain volume')
    template = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='template gca')
    out_file = File(argstr='%s', exists=False, name_source=['in_file'], name_template='%s_transform.lta', hash_files=False, keep_extension=False, position=-1, desc='output transform')
    skull = traits.Bool(argstr='-skull', desc='align to atlas containing skull (uns=5)')
    mask = File(argstr='-mask %s', exists=True, desc='use volume as a mask')
    nbrspacing = traits.Int(argstr='-uns %d', desc='align to atlas containing skull setting unknown_nbr_spacing = nbrspacing')
    transform = File(argstr='-t %s', exists=True, desc='Previously computed transform')