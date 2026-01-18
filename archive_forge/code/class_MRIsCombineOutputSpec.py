import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsCombineOutputSpec(TraitedSpec):
    """
    Uses Freesurfer's mris_convert to combine two surface files into one.
    """
    out_file = File(exists=True, desc='Output filename. Combined surfaces from in_files.')