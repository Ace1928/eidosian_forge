import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class BinarizeOutputSpec(TraitedSpec):
    binary_file = File(exists=True, desc='binarized output volume')
    count_file = File(desc='ascii file containing number of hits')