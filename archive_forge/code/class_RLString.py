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
class RLString(str):
    """allows specification of extra properties of a string using a dictionary of extra attributes
    eg fontName = RLString('proxima-nova-bold',
                    svgAttrs=dict(family='"proxima-nova"',weight='bold'))
    """

    def __new__(cls, v, **kwds):
        self = str.__new__(cls, v)
        for k, v in kwds.items():
            setattr(self, k, v)
        return self