import numpy as np
import nibabel as nb
from ... import logging
from ..base import TraitedSpec, File, isdefined
from .base import DipyDiffusionInterface, DipyBaseInterfaceInputSpec
class TensorModeOutputSpec(TraitedSpec):
    out_file = File(exists=True)