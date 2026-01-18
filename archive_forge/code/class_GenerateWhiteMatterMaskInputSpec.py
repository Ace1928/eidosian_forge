import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
class GenerateWhiteMatterMaskInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='Diffusion-weighted images')
    binary_mask = File(exists=True, argstr='%s', mandatory=True, position=-2, desc='Binary brain mask')
    out_WMProb_filename = File(genfile=True, argstr='%s', position=-1, desc='Output WM probability image filename')
    encoding_file = File(exists=True, argstr='-grad %s', mandatory=True, position=1, desc='Gradient encoding, supplied as a 4xN text file with each line is in the format [ X Y Z b ], where [ X Y Z ] describe the direction of the applied gradient, and b gives the b-value in units (1000 s/mm^2). See FSL2MRTrix')
    noise_level_margin = traits.Float(argstr='-margin %s', desc='Specify the width of the margin on either side of the image to be used to estimate the noise level (default = 10)')