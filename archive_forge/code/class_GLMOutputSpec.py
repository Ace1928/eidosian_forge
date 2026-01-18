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
class GLMOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='file name of GLM parameters (if generated)')
    out_cope = OutputMultiPath(File(exists=True), desc='output file name for COPEs (either as text file or image)')
    out_z = OutputMultiPath(File(exists=True), desc='output file name for COPEs (either as text file or image)')
    out_t = OutputMultiPath(File(exists=True), desc='output file name for t-stats (either as text file or image)')
    out_p = OutputMultiPath(File(exists=True), desc='output file name for p-values of Z-stats (either as text file or image)')
    out_f = OutputMultiPath(File(exists=True), desc='output file name for F-value of full model fit')
    out_pf = OutputMultiPath(File(exists=True), desc='output file name for p-value for full model fit')
    out_res = OutputMultiPath(File(exists=True), desc='output file name for residuals')
    out_varcb = OutputMultiPath(File(exists=True), desc='output file name for variance of COPEs')
    out_sigsq = OutputMultiPath(File(exists=True), desc='output file name for residual noise variance sigma-square')
    out_data = OutputMultiPath(File(exists=True), desc='output file for preprocessed data')
    out_vnscales = OutputMultiPath(File(exists=True), desc='output file name for scaling factors for variance normalisation')