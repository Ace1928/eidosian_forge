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
class Randomise(FSLCommand):
    """FSL Randomise: feeds the 4D projected FA data into GLM
    modelling and thresholding
    in order to find voxels which correlate with your model

    Example
    -------
    >>> import nipype.interfaces.fsl as fsl
    >>> rand = fsl.Randomise(in_file='allFA.nii', mask = 'mask.nii', tcon='design.con', design_mat='design.mat')
    >>> rand.cmdline
    'randomise -i allFA.nii -o "randomise" -d design.mat -t design.con -m mask.nii'

    """
    _cmd = 'randomise'
    input_spec = RandomiseInputSpec
    output_spec = RandomiseOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['tstat_files'] = glob(self._gen_fname('%s_tstat*.nii' % self.inputs.base_name))
        outputs['fstat_files'] = glob(self._gen_fname('%s_fstat*.nii' % self.inputs.base_name))
        prefix = False
        if self.inputs.tfce or self.inputs.tfce2D:
            prefix = 'tfce'
        elif self.inputs.vox_p_values:
            prefix = 'vox'
        elif self.inputs.c_thresh or self.inputs.f_c_thresh:
            prefix = 'clustere'
        elif self.inputs.cm_thresh or self.inputs.f_cm_thresh:
            prefix = 'clusterm'
        if prefix:
            outputs['t_p_files'] = glob(self._gen_fname('%s_%s_p_tstat*' % (self.inputs.base_name, prefix)))
            outputs['t_corrected_p_files'] = glob(self._gen_fname('%s_%s_corrp_tstat*.nii' % (self.inputs.base_name, prefix)))
            outputs['f_p_files'] = glob(self._gen_fname('%s_%s_p_fstat*.nii' % (self.inputs.base_name, prefix)))
            outputs['f_corrected_p_files'] = glob(self._gen_fname('%s_%s_corrp_fstat*.nii' % (self.inputs.base_name, prefix)))
        return outputs