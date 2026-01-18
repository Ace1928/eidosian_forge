import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsConvertOutputSpec(TraitedSpec):
    """
    Uses Freesurfer's mris_convert to convert surface files to various formats
    """
    converted = File(exists=True, desc='converted output surface')