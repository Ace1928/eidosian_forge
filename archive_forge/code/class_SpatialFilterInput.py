import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class SpatialFilterInput(KernelInput):
    operation = traits.Enum('mean', 'median', 'meanu', argstr='-f%s', position=6, mandatory=True, desc='operation to filter with')