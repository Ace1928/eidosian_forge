import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class CentralityInputSpec(AFNICommandInputSpec):
    """Common input spec class for all centrality-related commands"""
    mask = File(desc='mask file to mask input data', argstr='-mask %s', exists=True)
    thresh = traits.Float(desc='threshold to exclude connections where corr <= thresh', argstr='-thresh %f')
    polort = traits.Int(desc='', argstr='-polort %d')
    autoclip = traits.Bool(desc='Clip off low-intensity regions in the dataset', argstr='-autoclip')
    automask = traits.Bool(desc='Mask the dataset to target brain-only voxels', argstr='-automask')