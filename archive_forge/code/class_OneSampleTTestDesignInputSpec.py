import os
from glob import glob
import numpy as np
from ... import logging
from ...utils.filemanip import ensure_list, simplify_list, split_filename
from ..base import (
from .base import SPMCommand, SPMCommandInputSpec, scans_for_fnames, ImageFileSPM
class OneSampleTTestDesignInputSpec(FactorialDesignInputSpec):
    in_files = traits.List(File(exists=True), field='des.t1.scans', mandatory=True, minlen=2, desc='input files')