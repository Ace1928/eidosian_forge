import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging
def _is_4d(self):
    self._cmd = 'c4d' if self.inputs.is_4d else 'c3d'