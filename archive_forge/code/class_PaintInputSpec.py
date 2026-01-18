import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class PaintInputSpec(FSTraitedSpec):
    in_surf = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='Surface file with grid (vertices) onto which the ' + "template data is to be sampled or 'painted'")
    template = File(argstr='%s', exists=True, mandatory=True, position=-3, desc='Template file')
    template_param = traits.Int(desc='Frame number of the input template')
    averages = traits.Int(argstr='-a %d', desc='Average curvature patterns')
    out_file = File(argstr='%s', exists=False, position=-1, name_template='%s.avg_curv', hash_files=False, name_source=['in_surf'], keep_extension=False, desc='File containing a surface-worth of per-vertex values, ' + "saved in 'curvature' format.")