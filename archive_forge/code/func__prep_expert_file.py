import os
import os.path as op
from glob import glob
import shutil
import sys
import numpy as np
from nibabel import load
from ... import logging, LooseVersion
from ...utils.filemanip import fname_presuffix, check_depends
from ..io import FreeSurferSource
from ..base import (
from .base import FSCommand, FSTraitedSpec, FSTraitedSpecOpenMP, FSCommandOpenMP, Info
from .utils import copy2subjdir
def _prep_expert_file(self):
    if isdefined(self.inputs.expert):
        return ''
    lines = []
    for binary in self._binaries:
        args = getattr(self.inputs, binary)
        if isdefined(args):
            lines.append('{} {}\n'.format(binary, args))
    if lines == []:
        return ''
    contents = ''.join(lines)
    if not isdefined(self.inputs.xopts) and self._get_expert_file() == contents:
        return ' -xopts-use'
    expert_fname = os.path.abspath('expert.opts')
    with open(expert_fname, 'w') as fobj:
        fobj.write(contents)
    return ' -expert {}'.format(expert_fname)