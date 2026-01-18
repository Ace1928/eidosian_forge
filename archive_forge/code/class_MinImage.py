import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MinImage(MathsCommand):
    """Use fslmaths to generate a minimum image across a given dimension."""
    input_spec = MinImageInput
    _suffix = '_min'