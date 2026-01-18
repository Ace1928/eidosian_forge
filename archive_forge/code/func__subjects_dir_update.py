import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
def _subjects_dir_update(self):
    if self.inputs.subjects_dir:
        self.inputs.environ.update({'SUBJECTS_DIR': self.inputs.subjects_dir})