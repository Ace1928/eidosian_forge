import os
import warnings
from ...utils.filemanip import fname_presuffix, split_filename, copyfile
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class TractSkeletonOutputSpec(TraitedSpec):
    projected_data = File(desc='input data projected onto skeleton')
    skeleton_file = File(desc='tract skeleton image')