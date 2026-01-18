import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
class JSONFileSinkInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    out_file = File(desc='JSON sink file')
    in_dict = traits.Dict(value={}, usedefault=True, desc='input JSON dictionary')
    _outputs = traits.Dict(value={}, usedefault=True)

    def __setattr__(self, key, value):
        if key not in self.copyable_trait_names():
            if not isdefined(value):
                super(JSONFileSinkInputSpec, self).__setattr__(key, value)
            self._outputs[key] = value
        else:
            if key in self._outputs:
                self._outputs[key] = value
            super(JSONFileSinkInputSpec, self).__setattr__(key, value)