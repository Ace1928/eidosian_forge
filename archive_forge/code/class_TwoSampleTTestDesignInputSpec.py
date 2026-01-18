import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class TwoSampleTTestDesignInputSpec(FactorialDesignInputSpec):
    group1_files = traits.List(File(exists=True), field='des.t2.scans1', mandatory=True, minlen=2, desc='Group 1 input files')
    group2_files = traits.List(File(exists=True), field='des.t2.scans2', mandatory=True, minlen=2, desc='Group 2 input files')
    dependent = traits.Bool(field='des.t2.dept', desc='Are the measurements dependent between levels')
    unequal_variance = traits.Bool(field='des.t2.variance', desc='Are the variances equal or unequal between groups')