import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
@classmethod
def _get_distver_from_lsb_release(cls):
    lsb_release = shutil.which('lsb_release')
    dist = subprocess.run([lsb_release, '-si'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    ver = subprocess.run([lsb_release, '-sr'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return (cls._map_linux_dist(dist.stdout), ver.stdout.strip())