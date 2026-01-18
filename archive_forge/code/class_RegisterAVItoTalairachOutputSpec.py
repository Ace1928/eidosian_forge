import os
import os.path
from ... import logging
from ...utils.filemanip import split_filename, copyfile
from .base import (
from ..base import isdefined, TraitedSpec, File, traits, Directory
class RegisterAVItoTalairachOutputSpec(FSScriptOutputSpec):
    out_file = File(exists=False, desc='The output file for RegisterAVItoTalairach')