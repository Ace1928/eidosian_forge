import os
import nibabel as nb
import numpy as np
from ...utils.filemanip import split_filename, fname_presuffix
from .base import NipyBaseInterface, have_nipy
from ..base import (
class SpaceTimeRealignerOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True), desc='Realigned files')
    par_file = OutputMultiPath(File(exists=True), desc='Motion parameter files. Angles are not euler angles')