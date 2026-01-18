import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class AddCSVRowInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    in_file = File(mandatory=True, desc='Input comma-separated value (CSV) files')
    _outputs = traits.Dict(traits.Any, value={}, usedefault=True)

    def __setattr__(self, key, value):
        if key not in self.copyable_trait_names():
            if not isdefined(value):
                super(AddCSVRowInputSpec, self).__setattr__(key, value)
            self._outputs[key] = value
        else:
            if key in self._outputs:
                self._outputs[key] = value
            super(AddCSVRowInputSpec, self).__setattr__(key, value)