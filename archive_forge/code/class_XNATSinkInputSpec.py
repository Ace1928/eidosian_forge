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
class XNATSinkInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    _outputs = traits.Dict(Str, value={}, usedefault=True)
    server = Str(mandatory=True, requires=['user', 'pwd'], xor=['config'])
    user = Str()
    pwd = traits.Password()
    config = File(mandatory=True, xor=['server'])
    cache_dir = Directory(desc='')
    project_id = Str(desc='Project in which to store the outputs', mandatory=True)
    subject_id = Str(desc='Set to subject id', mandatory=True)
    experiment_id = Str(desc='Set to workflow name', mandatory=True)
    assessor_id = Str(desc='Option to customize outputs representation in XNAT - assessor level will be used with specified id', xor=['reconstruction_id'])
    reconstruction_id = Str(desc='Option to customize outputs representation in XNAT - reconstruction level will be used with specified id', xor=['assessor_id'])
    share = traits.Bool(False, desc='Option to share the subjects from the original projectinstead of creating new ones when possible - the created experiments are then shared back to the original project', usedefault=True)

    def __setattr__(self, key, value):
        if key not in self.copyable_trait_names():
            self._outputs[key] = value
        else:
            super(XNATSinkInputSpec, self).__setattr__(key, value)