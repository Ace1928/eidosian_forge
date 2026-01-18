import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class ApplyMaskInput(MathsInput):
    mask_file = File(exists=True, mandatory=True, argstr='-mas %s', position=4, desc='binary image defining mask space')