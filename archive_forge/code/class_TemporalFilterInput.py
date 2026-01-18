import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class TemporalFilterInput(MathsInput):
    lowpass_sigma = traits.Float(-1, argstr='%.6f', position=5, usedefault=True, desc='lowpass filter sigma (in volumes)')
    highpass_sigma = traits.Float(-1, argstr='-bptf %.6f', position=4, usedefault=True, desc='highpass filter sigma (in volumes)')