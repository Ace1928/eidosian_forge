import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class CreateWarpedInputSpec(SPMCommandInputSpec):
    image_files = InputMultiPath(ImageFileSPM(exists=True), mandatory=True, desc='A list of files to be warped', field='crt_warped.images', copyfile=False)
    flowfield_files = InputMultiPath(ImageFileSPM(exists=True), copyfile=False, desc='DARTEL flow fields u_rc1*', field='crt_warped.flowfields', mandatory=True)
    iterations = traits.Range(low=0, high=9, desc='The number of iterations: log2(number of time steps)', field='crt_warped.K')
    interp = traits.Range(low=0, high=7, field='crt_warped.interp', desc='degree of b-spline used for interpolation')
    modulate = traits.Bool(field='crt_warped.jactransf', desc='Modulate images')