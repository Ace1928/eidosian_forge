import os
from copy import deepcopy
import numpy as np
from ...utils.filemanip import (
from ..base import (
from .base import (
class CoregisterOutputSpec(TraitedSpec):
    coregistered_source = OutputMultiPath(File(exists=True), desc='Coregistered source files')
    coregistered_files = OutputMultiPath(File(exists=True), desc='Coregistered other files')