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
def _dump(self, f):
    try:
        pos = f.tell()
        pickle.dump(self.store, f)
    except:
        S = self.store.copy()
        ff = BytesIO()
        for k, v in S.items():
            try:
                pickle.dump({k: v}, ff)
            except:
                S[k] = '<unpicklable object %r>' % v
        f.seek(pos, 0)
        pickle.dump(S, f)