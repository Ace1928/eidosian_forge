import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class Text2VestInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, desc='plain text file representing your design, contrast, or f-test matrix', argstr='%s', position=0)
    out_file = File(mandatory=True, desc='file name to store matrix data in the format used by FSL tools (e.g., design.mat, design.con design.fts)', argstr='%s', position=1)