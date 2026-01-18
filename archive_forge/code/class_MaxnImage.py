import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MaxnImage(MathsCommand):
    """Use fslmaths to generate an image of index of max across
    a given dimension.

    """
    input_spec = MaxnImageInput
    _suffix = '_maxn'