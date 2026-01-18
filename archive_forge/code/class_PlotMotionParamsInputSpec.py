import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class PlotMotionParamsInputSpec(FSLCommandInputSpec):
    in_file = traits.Either(File(exists=True), traits.List(File(exists=True)), mandatory=True, argstr='%s', position=1, desc='file with motion parameters')
    in_source = traits.Enum('spm', 'fsl', mandatory=True, desc='which program generated the motion parameter file - fsl, spm')
    plot_type = traits.Enum('rotations', 'translations', 'displacement', argstr='%s', mandatory=True, desc='which motion type to plot - rotations, translations, displacement')
    plot_size = traits.Tuple(traits.Int, traits.Int, argstr='%s', desc='plot image height and width')
    out_file = File(argstr='-o %s', genfile=True, desc='image to write', hash_files=False)