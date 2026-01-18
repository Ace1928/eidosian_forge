import os
import re
import sys
import shlex
import copy
from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError
from numpy.distutils import log
from numpy.distutils.misc_util import (
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file
def _build_npy_pkg_config(self, info, gd):
    template, install_dir, subst_dict = info
    template_dir = os.path.dirname(template)
    for k, v in gd.items():
        subst_dict[k] = v
    if self.inplace == 1:
        generated_dir = os.path.join(template_dir, install_dir)
    else:
        generated_dir = os.path.join(self.build_src, template_dir, install_dir)
    generated = os.path.basename(os.path.splitext(template)[0])
    generated_path = os.path.join(generated_dir, generated)
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    subst_vars(generated_path, template, subst_dict)
    full_install_dir = os.path.join(template_dir, install_dir)
    return (full_install_dir, generated_path)