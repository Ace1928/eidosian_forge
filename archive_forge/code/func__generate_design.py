from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
def _generate_design(self, infolist=None):
    if isdefined(self.inputs.subject_info):
        infolist = self.inputs.subject_info
    else:
        infolist = gen_info(self.inputs.event_files)
    sparselist = self._generate_clustered_design(infolist)
    super(SpecifySparseModel, self)._generate_design(infolist=sparselist)