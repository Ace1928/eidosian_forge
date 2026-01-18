import os
import nibabel as nb
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import split_filename
class ActivationCountInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='input file, generally a list of z-stat maps')
    threshold = traits.Float(mandatory=True, desc='binarization threshold. E.g. a threshold of 1.65 corresponds to a two-sided Z-test of p<.10')