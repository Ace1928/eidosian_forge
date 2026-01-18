import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class MPRtoMNI305InputSpec(FSTraitedSpec):
    reference_dir = Directory('', exists=True, mandatory=True, usedefault=True, desc='TODO')
    target = traits.String('', mandatory=True, usedefault=True, desc='input atlas file')
    in_file = File(argstr='%s', usedefault=True, desc='the input file prefix for MPRtoMNI305')