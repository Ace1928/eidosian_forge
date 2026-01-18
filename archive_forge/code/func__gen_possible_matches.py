import os
import platform
import itertools
from .xmltodict import parse as xmltodictparser
import subprocess as sp
import numpy as np
from .edge import canny
from .stpyr import SpatialSteerablePyramid, rolling_window
from .mscn import compute_image_mscn_transform, gen_gauss_window
from .stats import ggd_features, aggd_features, paired_product
def _gen_possible_matches(filename):
    path_parts = os.environ.get('PATH', '').split(os.pathsep)
    path_parts = itertools.chain((os.curdir,), path_parts)
    possible_paths = map(lambda path_part: os.path.join(path_part, filename), path_parts)
    if platform.system() == 'Windows':
        possible_paths = imapchain(lambda path: (path, path + '.bat', path + '.com', path + '.exe'), possible_paths)
    possible_paths = map(os.path.abspath, possible_paths)
    result = iter_unique(possible_paths)
    return result