import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SUSANOutputSpec(TraitedSpec):
    smoothed_file = File(exists=True, desc='smoothed output file')