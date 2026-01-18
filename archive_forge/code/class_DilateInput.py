import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class DilateInput(KernelInput):
    operation = traits.Enum('mean', 'modal', 'max', argstr='-dil%s', position=6, mandatory=True, desc='filtering operation to perform in dilation')