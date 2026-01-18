import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class ErrorMapOutputSpec(TraitedSpec):
    out_map = File(exists=True, desc='resulting error map')
    distance = traits.Float(desc='Average distance between volume 1 and 2')