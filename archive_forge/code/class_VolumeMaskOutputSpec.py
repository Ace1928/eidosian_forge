import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class VolumeMaskOutputSpec(TraitedSpec):
    out_ribbon = File(exists=False, desc='Output cortical ribbon mask')
    lh_ribbon = File(exists=False, desc='Output left cortical ribbon mask')
    rh_ribbon = File(exists=False, desc='Output right cortical ribbon mask')