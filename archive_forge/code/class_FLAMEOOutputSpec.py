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
class FLAMEOOutputSpec(TraitedSpec):
    pes = OutputMultiPath(File(exists=True), desc='Parameter estimates for each column of the design matrix for each voxel')
    res4d = OutputMultiPath(File(exists=True), desc='Model fit residual mean-squared error for each time point')
    copes = OutputMultiPath(File(exists=True), desc='Contrast estimates for each contrast')
    var_copes = OutputMultiPath(File(exists=True), desc='Variance estimates for each contrast')
    zstats = OutputMultiPath(File(exists=True), desc='z-stat file for each contrast')
    tstats = OutputMultiPath(File(exists=True), desc='t-stat file for each contrast')
    zfstats = OutputMultiPath(File(exists=True), desc='z stat file for each f contrast')
    fstats = OutputMultiPath(File(exists=True), desc='f-stat file for each contrast')
    mrefvars = OutputMultiPath(File(exists=True), desc='mean random effect variances for each contrast')
    tdof = OutputMultiPath(File(exists=True), desc='temporal dof file for each contrast')
    weights = OutputMultiPath(File(exists=True), desc='weights file for each contrast')
    stats_dir = Directory(File(exists=True), desc='directory storing model estimation output')