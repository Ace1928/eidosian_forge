import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FIRSTInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, position=-2, copyfile=False, argstr='-i %s', desc='input data file')
    out_file = File('segmented', usedefault=True, mandatory=True, position=-1, argstr='-o %s', desc='output data file', hash_files=False)
    verbose = traits.Bool(argstr='-v', position=1, desc='Use verbose logging.')
    brain_extracted = traits.Bool(argstr='-b', position=2, desc='Input structural image is already brain-extracted')
    no_cleanup = traits.Bool(argstr='-d', position=3, desc='Input structural image is already brain-extracted')
    method = traits.Enum('auto', 'fast', 'none', xor=['method_as_numerical_threshold'], argstr='-m %s', position=4, usedefault=True, desc="Method must be one of auto, fast, none, or it can be entered using the 'method_as_numerical_threshold' input")
    method_as_numerical_threshold = traits.Float(argstr='-m %.4f', position=4, desc="Specify a numerical threshold value or use the 'method' input to choose auto, fast, or none")
    list_of_specific_structures = traits.List(traits.Str, argstr='-s %s', sep=',', position=5, minlen=1, desc='Runs only on the specified structures (e.g. L_Hipp, R_HippL_Accu, R_Accu, L_Amyg, R_AmygL_Caud, R_Caud, L_Pall, R_PallL_Puta, R_Puta, L_Thal, R_Thal, BrStem')
    affine_file = File(exists=True, position=6, argstr='-a %s', desc='Affine matrix to use (e.g. img2std.mat) (does not re-run registration)')