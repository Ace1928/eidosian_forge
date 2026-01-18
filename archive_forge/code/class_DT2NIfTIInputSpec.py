import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class DT2NIfTIInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='-inputfile %s', mandatory=True, position=1, desc='tract file')
    output_root = File(argstr='-outputroot %s', position=2, genfile=True, desc='filename root prepended onto the names of three output files.')
    header_file = File(exists=True, argstr='-header %s', mandatory=True, position=3, desc=' A Nifti .nii or .hdr file containing the header information')