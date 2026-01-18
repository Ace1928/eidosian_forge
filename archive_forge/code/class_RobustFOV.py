import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class RobustFOV(FSLCommand):
    """Automatically crops an image removing lower head and neck.

    Interface is stable 5.0.0 to 5.0.9, but default brainsize changed from
    150mm to 170mm.
    """
    _cmd = 'robustfov'
    input_spec = RobustFOVInputSpec
    output_spec = RobustFOVOutputSpec