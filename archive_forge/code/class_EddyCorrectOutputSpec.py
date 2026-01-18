import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class EddyCorrectOutputSpec(TraitedSpec):
    eddy_corrected = File(exists=True, desc='path/name of 4D eddy corrected output file')