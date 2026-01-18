from __future__ import print_function, absolute_import, division
from . import __version__
from .report import report
import os
import importlib
import inspect
import argparse
import distutils.dir_util
import shutil
from collections import OrderedDict
import glob
import sys
import tarfile
import time
import zipfile
import yaml
class DirectoryContext(object):
    """
    Context Manager for changing directories
    """

    def __init__(self, path):
        self.old_dir = os.getcwd()
        self.new_dir = path

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)