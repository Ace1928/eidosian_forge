import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class DWI2SphericalHarmonicsImageOutputSpec(TraitedSpec):
    spherical_harmonics_image = File(exists=True, desc='Spherical harmonics image')