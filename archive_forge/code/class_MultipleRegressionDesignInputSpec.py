import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class MultipleRegressionDesignInputSpec(FactorialDesignInputSpec):
    in_files = traits.List(File(exists=True), field='des.mreg.scans', mandatory=True, minlen=2, desc='List of files')
    include_intercept = traits.Bool(True, field='des.mreg.incint', usedefault=True, desc='Include intercept in design')
    user_covariates = InputMultiPath(traits.Dict(key_trait=traits.Enum('vector', 'name', 'centering')), field='des.mreg.mcov', desc='covariate dictionary {vector, name, centering}')