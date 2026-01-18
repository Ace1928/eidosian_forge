import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class MaxImage(MathsCommand):
    """Use fslmaths to generate a max image across a given dimension.

    Examples
    --------
    >>> from nipype.interfaces.fsl.maths import MaxImage
    >>> maxer = MaxImage()
    >>> maxer.inputs.in_file = "functional.nii"  # doctest: +SKIP
    >>> maxer.dimension = "T"
    >>> maxer.cmdline  # doctest: +SKIP
    'fslmaths functional.nii -Tmax functional_max.nii'

    """
    input_spec = MaxImageInput
    _suffix = '_max'