import configparser
import glob
import os
import sys
from os.path import join as pjoin
from packaging.version import Version
from .environment import get_nipy_system_dir, get_nipy_user_dir
class Bomber:
    """Class to raise an informative error when used"""

    def __init__(self, name, msg):
        self.name = name
        self.msg = msg

    def __getattr__(self, attr_name):
        """Raise informative error accessing not-found attributes"""
        raise BomberError(f'Trying to access attribute "{attr_name}" of non-existent data "{self.name}"\n\n{self.msg}\n')