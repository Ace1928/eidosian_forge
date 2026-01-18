from contextlib import contextmanager
import glob
from importlib import import_module
import io
import itertools
import os.path as osp
import re
import sys
import warnings
import zipfile
import configparser
class CaseSensitiveConfigParser(configparser.ConfigParser):
    optionxform = staticmethod(str)