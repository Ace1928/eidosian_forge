from copy import deepcopy
import csv, math, os
from nibabel import load
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list
from ..utils.misc import normalize_mc_params
from .. import config, logging
class SpecifySparseModelOutputSpec(SpecifyModelOutputSpec):
    sparse_png_file = File(desc='PNG file showing sparse design')
    sparse_svg_file = File(desc='SVG file showing sparse design')