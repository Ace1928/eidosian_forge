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
class FEATModelOutpuSpec(TraitedSpec):
    design_file = File(exists=True, desc='Mat file containing ascii matrix for design')
    design_image = File(exists=True, desc='Graphical representation of design matrix')
    design_cov = File(exists=True, desc='Graphical representation of design covariance')
    con_file = File(exists=True, desc='Contrast file containing contrast vectors')
    fcon_file = File(desc='Contrast file containing contrast vectors')