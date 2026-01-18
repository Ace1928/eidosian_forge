import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class SmoothEstimateOutputSpec(TraitedSpec):
    dlh = traits.Float(desc='smoothness estimate sqrt(det(Lambda))')
    volume = traits.Int(desc='number of voxels in mask')
    resels = traits.Float(desc='volume of resel, in voxels, defined as FWHM_x * FWHM_y * FWHM_z')