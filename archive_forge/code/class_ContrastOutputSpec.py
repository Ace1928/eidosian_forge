import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class ContrastOutputSpec(TraitedSpec):
    out_contrast = File(exists=False, desc='Output contrast file from Contrast')
    out_stats = File(exists=False, desc='Output stats file from Contrast')
    out_log = File(exists=True, desc='Output log from Contrast')