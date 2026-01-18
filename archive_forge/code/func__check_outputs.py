from copy import deepcopy
import re
import numpy as np
from ... import config
from ...interfaces.base import DynamicTraitedSpec
from ...utils.filemanip import loadpkl, savepkl
def _check_outputs(self, parameter):
    return hasattr(self.outputs, parameter)