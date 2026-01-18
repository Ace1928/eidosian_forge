import os.path as op
import numpy as np
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
class FSL2MRTrixInputSpec(TraitedSpec):
    bvec_file = File(exists=True, mandatory=True, desc='FSL b-vectors file (3xN text file)')
    bval_file = File(exists=True, mandatory=True, desc='FSL b-values file (1xN text file)')
    invert_x = traits.Bool(False, usedefault=True, desc='Inverts the b-vectors along the x-axis')
    invert_y = traits.Bool(False, usedefault=True, desc='Inverts the b-vectors along the y-axis')
    invert_z = traits.Bool(False, usedefault=True, desc='Inverts the b-vectors along the z-axis')
    out_encoding_file = File(genfile=True, desc='Output encoding filename')