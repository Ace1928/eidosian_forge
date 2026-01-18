import os.path as op
import numpy as np
import nibabel as nb
from looseversion import LooseVersion
from ... import logging
from ..base import TraitedSpec, File, traits, isdefined
from .base import (
class EstimateResponseSHOutputSpec(TraitedSpec):
    response = File(exists=True, desc='the response file')
    out_mask = File(exists=True, desc='output wm mask')