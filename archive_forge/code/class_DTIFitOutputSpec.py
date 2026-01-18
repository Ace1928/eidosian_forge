import os
from ...utils.filemanip import split_filename
from ..base import (
class DTIFitOutputSpec(TraitedSpec):
    tensor_fitted = File(exists=True, desc='path/name of 4D volume in voxel order')