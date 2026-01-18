import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class SpatialFilter(MathsCommand):
    """Use fslmaths to spatially filter an image."""
    input_spec = SpatialFilterInput
    _suffix = '_filt'