import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class BlurOutputSpec(TraitedSpec):
    output_file = File(desc='Blurred output file.', exists=True)
    gradient_dxyz = File(desc='Gradient dxyz.')
    partial_dx = File(desc='Partial gradient dx.')
    partial_dy = File(desc='Partial gradient dy.')
    partial_dz = File(desc='Partial gradient dz.')
    partial_dxyz = File(desc='Partial gradient dxyz.')