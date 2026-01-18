import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class ConcatenateOutputSpec(TraitedSpec):
    concatenated_file = File(exists=True, desc='Path/name of the output volume')