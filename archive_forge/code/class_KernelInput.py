import os
import numpy as np
from ..base import TraitedSpec, File, traits, InputMultiPath, isdefined
from .base import FSLCommand, FSLCommandInputSpec
class KernelInput(MathsInput):
    kernel_shape = traits.Enum('3D', '2D', 'box', 'boxv', 'gauss', 'sphere', 'file', argstr='-kernel %s', position=4, desc='kernel shape to use')
    kernel_size = traits.Float(argstr='%.4f', position=5, xor=['kernel_file'], desc='kernel size - voxels for box/boxv, mm for sphere, mm sigma for gauss')
    kernel_file = File(exists=True, argstr='%s', position=5, xor=['kernel_size'], desc='use external file for kernel')