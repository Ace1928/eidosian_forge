import os, pickle, sys, time, types, datetime, importlib
from ast import literal_eval
from base64 import decodebytes as base64_decodebytes, encodebytes as base64_encodebytes
from io import BytesIO
from hashlib import md5
from reportlab.lib.rltempfile import get_rl_tempfile, get_rl_tempdir
from . rl_safe_eval import rl_safe_exec, rl_safe_eval, safer_globals, rl_extended_literal_eval
from PIL import Image
import builtins
import reportlab
import glob, fnmatch
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from importlib import util as importlib_util
import itertools
def __rl_get_module__(name, dir):
    for ext in ('.py', '.pyw', '.pyo', '.pyc', '.pyd'):
        path = os.path.join(dir, name + ext)
        if os.path.isfile(path):
            spec = importlib_util.spec_from_file_location(name, path)
            return spec.loader.load_module()
    raise ImportError('no suitable file found')