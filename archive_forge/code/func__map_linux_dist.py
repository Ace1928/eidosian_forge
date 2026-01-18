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
def _map_linux_dist(cls, dist):
    dist = dist.lower().strip().replace(' ', '')
    _map = [('redhat', 'rhel'), 'fedora', 'ubuntu', 'debian', 'centos', 'rocky', 'almalinux', 'scientific', 'linuxmint']
    for key in _map:
        if key.__class__ is tuple:
            key, val = key
        else:
            val = key
        if key in dist:
            return val
    return dist