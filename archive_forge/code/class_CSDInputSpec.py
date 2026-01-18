import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class CSDInputSpec(DipyBaseInterfaceInputSpec):
    in_mask = File(exists=True, desc='input mask in which compute tensors')
    response = File(exists=True, desc='single fiber estimated response')
    sh_order = traits.Int(8, usedefault=True, desc='maximal shperical harmonics order')
    save_fods = traits.Bool(True, usedefault=True, desc='save fODFs in file')
    out_fods = File(desc='fODFs output file name')