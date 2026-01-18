import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MinImageInput(MathsInput):
    dimension = traits.Enum('T', 'X', 'Y', 'Z', usedefault=True, argstr='-%smin', position=4, desc='dimension to min across')