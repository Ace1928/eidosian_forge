import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class RegisterInputSpec(FSTraitedSpec):
    in_surf = File(argstr='%s', exists=True, mandatory=True, position=-3, copyfile=True, desc='Surface to register, often {hemi}.sphere')
    target = File(argstr='%s', exists=True, mandatory=True, position=-2, desc='The data to register to. In normal recon-all usage, ' + 'this is a template file for average surface.')
    in_sulc = File(exists=True, mandatory=True, copyfile=True, desc='Undocumented mandatory input file ${SUBJECTS_DIR}/surf/{hemisphere}.sulc ')
    out_file = File(argstr='%s', exists=False, position=-1, genfile=True, desc='Output surface file to capture registration')
    curv = traits.Bool(argstr='-curv', requires=['in_smoothwm'], desc='Use smoothwm curvature for final alignment')
    in_smoothwm = File(exists=True, copyfile=True, desc='Undocumented input file ${SUBJECTS_DIR}/surf/{hemisphere}.smoothwm ')