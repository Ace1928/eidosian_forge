import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2LabelOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Output label')