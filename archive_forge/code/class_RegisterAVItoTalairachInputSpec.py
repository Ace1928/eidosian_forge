import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class RegisterAVItoTalairachInputSpec(FSTraitedSpec):
    in_file = File(argstr='%s', exists=True, mandatory=True, position=0, desc='The input file')
    target = File(argstr='%s', exists=True, mandatory=True, position=1, desc='The target file')
    vox2vox = File(argstr='%s', exists=True, mandatory=True, position=2, desc='The vox2vox file')
    out_file = File('talairach.auto.xfm', usedefault=True, argstr='%s', position=3, desc='The transform output')