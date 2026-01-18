import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsCombineInputSpec(FSTraitedSpec):
    """
    Uses Freesurfer's mris_convert to combine two surface files into one.
    """
    in_files = traits.List(File(Exists=True), maxlen=2, minlen=2, mandatory=True, position=1, argstr='--combinesurfs %s', desc='Two surfaces to be combined.')
    out_file = File(argstr='%s', position=-1, genfile=True, mandatory=True, desc='Output filename. Combined surfaces from in_files.')