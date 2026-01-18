import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class RealignOutputSpec(TraitedSpec):
    mean_image = File(exists=True, desc='Mean image file from the realignment')
    modified_in_files = OutputMultiPath(traits.Either(traits.List(File(exists=True)), File(exists=True)), desc='Copies of all files passed to in_files. Headers will have been modified to align all images with the first, or optionally to first do that, extract a mean image, and re-align to that mean image.')
    realigned_files = OutputMultiPath(traits.Either(traits.List(File(exists=True)), File(exists=True)), desc='If jobtype is write or estwrite, these will be the resliced files. Otherwise, they will be copies of in_files that have had their headers rewritten.')
    realignment_parameters = OutputMultiPath(File(exists=True), desc='Estimated translation and rotation parameters')