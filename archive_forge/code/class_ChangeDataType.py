import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class ChangeDataType(MathsCommand):
    """Use fslmaths to change the datatype of an image."""
    input_spec = ChangeDataTypeInput
    _suffix = '_chdt'