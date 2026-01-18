import os
import numpy as np
import nibabel as nb
import warnings
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import traits, TraitedSpec, InputMultiPath, File, isdefined
from .base import FSLCommand, FSLCommandInputSpec, Info
class ApplyTOPUPOutputSpec(TraitedSpec):
    out_corrected = File(exists=True, desc='name of 4D image file with unwarped images')