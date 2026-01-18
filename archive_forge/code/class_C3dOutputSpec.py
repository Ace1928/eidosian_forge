import os
from glob import glob
from .base import (
from ..utils.filemanip import split_filename
from .. import logging
class C3dOutputSpec(TraitedSpec):
    out_files = OutputMultiPath(File(exists=False))