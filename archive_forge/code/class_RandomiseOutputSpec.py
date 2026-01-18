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
class RandomiseOutputSpec(TraitedSpec):
    tstat_files = traits.List(File(exists=True), desc='t contrast raw statistic')
    fstat_files = traits.List(File(exists=True), desc='f contrast raw statistic')
    t_p_files = traits.List(File(exists=True), desc='f contrast uncorrected p values files')
    f_p_files = traits.List(File(exists=True), desc='f contrast uncorrected p values files')
    t_corrected_p_files = traits.List(File(exists=True), desc='t contrast FWE (Family-wise error) corrected p values files')
    f_corrected_p_files = traits.List(File(exists=True), desc='f contrast FWE (Family-wise error) corrected p values files')