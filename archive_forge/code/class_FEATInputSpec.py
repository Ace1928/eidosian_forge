import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class FEATInputSpec(FSLCommandInputSpec):
    fsf_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='File specifying the feat design spec file')