import configparser
import glob
import os
import sys
from os.path import join as pjoin
from packaging.version import Version
from .environment import get_nipy_system_dir, get_nipy_user_dir
def _cfg_value(fname, section='DATA', value='path'):
    """Utility function to fetch value from config file"""
    configp = configparser.ConfigParser()
    readfiles = configp.read(fname)
    if not readfiles:
        return ''
    try:
        return configp.get(section, value)
    except configparser.Error:
        return ''