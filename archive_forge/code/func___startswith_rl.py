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
def __startswith_rl(fn, _archivepfx=_archivepfx, _archivedirpfx=_archivedirpfx, _archive=_archive, _archivedir=_archivedir, os_path_normpath=os.path.normpath, os_path_normcase=os.path.normcase, os_getcwd=os.getcwd, os_sep=os.sep, os_sep_len=len(os.sep)):
    """if the name starts with a known prefix strip it off"""
    fn = os_path_normpath(fn.replace('/', os_sep))
    nfn = os_path_normcase(fn)
    if nfn in (_archivedir, _archive):
        return (1, '')
    if nfn.startswith(_archivepfx):
        return (1, fn[_archivepfxlen:])
    if nfn.startswith(_archivedirpfx):
        return (1, fn[_archivedirpfxlen:])
    cwd = os_path_normcase(os_getcwd())
    n = len(cwd)
    if nfn.startswith(cwd):
        if fn[n:].startswith(os_sep):
            return (1, fn[n + os_sep_len:])
        if n == len(fn):
            return (1, '')
    return (not os.path.isabs(fn), fn)