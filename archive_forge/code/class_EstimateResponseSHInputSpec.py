import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class EstimateResponseSHInputSpec(DipyBaseInterfaceInputSpec):
    in_evals = File(exists=True, mandatory=True, desc='input eigenvalues file')
    in_mask = File(exists=True, desc='input mask in which we find single fibers')
    fa_thresh = traits.Float(0.7, usedefault=True, desc='FA threshold')
    roi_radius = traits.Int(10, usedefault=True, desc='ROI radius to be used in auto_response')
    auto = traits.Bool(xor=['recursive'], desc='use the auto_response estimator from dipy')
    recursive = traits.Bool(xor=['auto'], desc='use the recursive response estimator from dipy')
    response = File('response.txt', usedefault=True, desc='the output response file')
    out_mask = File('wm_mask.nii.gz', usedefault=True, desc='computed wm mask')