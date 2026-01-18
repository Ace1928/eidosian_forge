import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class InvWarpOutputSpec(TraitedSpec):
    inverse_warp = File(exists=True, desc='Name of output file, containing warps that are the "reverse" of those in --warp.')