import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class MPRtoMNI305OutputSpec(FSScriptOutputSpec):
    out_file = File(exists=False, desc="The output file '<in_file>_to_<target>_t4_vox2vox.txt'")