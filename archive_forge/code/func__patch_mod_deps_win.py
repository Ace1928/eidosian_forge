import sys
import shutil
from getopt import getopt, GetoptError
import os
from os import environ, mkdir
from os.path import dirname, join, basename, exists, expanduser
import pkgutil
import re
import importlib
from kivy.logger import Logger, LOG_LEVELS
from kivy.utils import platform
from kivy._version import __version__, RELEASE as _KIVY_RELEASE, \
from kivy.logger import file_log_handler
def _patch_mod_deps_win(dep_mod, mod_name):
    import site
    dep_bins = []
    for d in [sys.prefix, site.USER_BASE]:
        p = join(d, 'share', mod_name, 'bin')
        if os.path.isdir(p):
            os.environ['PATH'] = p + os.pathsep + os.environ['PATH']
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(p)
            dep_bins.append(p)
    dep_mod.dep_bins = dep_bins