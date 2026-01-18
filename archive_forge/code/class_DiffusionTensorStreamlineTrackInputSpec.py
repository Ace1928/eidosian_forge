import os
import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class DiffusionTensorStreamlineTrackInputSpec(StreamlineTrackInputSpec):
    gradient_encoding_file = File(exists=True, argstr='-grad %s', mandatory=True, position=-2, desc='Gradient encoding, supplied as a 4xN text file with each line is in the format [ X Y Z b ], where [ X Y Z ] describe the direction of the applied gradient, and b gives the b-value in units (1000 s/mm^2). See FSL2MRTrix')