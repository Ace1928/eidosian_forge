import numpy as np
import nibabel as nb
from ... import logging
from ..base import TraitedSpec, File, isdefined
from .base import DipyDiffusionInterface, DipyBaseInterfaceInputSpec
class DTIInputSpec(DipyBaseInterfaceInputSpec):
    mask_file = File(exists=True, desc='An optional white matter mask')