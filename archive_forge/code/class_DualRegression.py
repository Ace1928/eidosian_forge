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
class DualRegression(FSLCommand):
    """Wrapper Script for Dual Regression Workflow

    Examples
    --------

    >>> dual_regression = DualRegression()
    >>> dual_regression.inputs.in_files = ["functional.nii", "functional2.nii", "functional3.nii"]
    >>> dual_regression.inputs.group_IC_maps_4D = "allFA.nii"
    >>> dual_regression.inputs.des_norm = False
    >>> dual_regression.inputs.one_sample_group_mean = True
    >>> dual_regression.inputs.n_perm = 10
    >>> dual_regression.inputs.out_dir = "my_output_directory"
    >>> dual_regression.cmdline
    'dual_regression allFA.nii 0 -1 10 my_output_directory functional.nii functional2.nii functional3.nii'
    >>> dual_regression.run() # doctest: +SKIP

    """
    input_spec = DualRegressionInputSpec
    output_spec = DualRegressionOutputSpec
    _cmd = 'dual_regression'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_dir):
            outputs['out_dir'] = os.path.abspath(self.inputs.out_dir)
        else:
            outputs['out_dir'] = self._gen_filename('out_dir')
        return outputs

    def _gen_filename(self, name):
        if name == 'out_dir':
            return os.getcwd()