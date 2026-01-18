import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class Label2VolOutputSpec(TraitedSpec):
    vol_label_file = File(exists=True, desc='output volume')