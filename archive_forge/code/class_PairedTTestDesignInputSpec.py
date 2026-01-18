import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class PairedTTestDesignInputSpec(FactorialDesignInputSpec):
    paired_files = traits.List(traits.List(File(exists=True), minlen=2, maxlen=2), field='des.pt.pair', mandatory=True, minlen=2, desc='List of paired files')
    grand_mean_scaling = traits.Bool(field='des.pt.gmsca', desc='Perform grand mean scaling')
    ancova = traits.Bool(field='des.pt.ancova', desc='Specify ancova-by-factor regressors')