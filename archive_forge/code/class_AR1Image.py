import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class AR1Image(MathsCommand):
    """Use fslmaths to generate an AR1 coefficient image across a
    given dimension. (Should use -odt float and probably demean first)

    """
    input_spec = AR1ImageInput
    _suffix = '_ar1'