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
def _findFiles(dirList, ext='.ttf'):
    from os.path import isfile, isdir, join as path_join
    from os import listdir
    ext = ext.lower()
    R = []
    A = R.append
    for D in dirList:
        if not isdir(D):
            continue
        for fn in listdir(D):
            fn = path_join(D, fn)
            if isfile(fn) and (not ext or fn.lower().endswith(ext)):
                A(fn)
    return R