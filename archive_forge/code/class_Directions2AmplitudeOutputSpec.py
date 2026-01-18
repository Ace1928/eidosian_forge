import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class Directions2AmplitudeOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='amplitudes image')