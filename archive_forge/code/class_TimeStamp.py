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
class TimeStamp:

    def __init__(self, invariant=None):
        if invariant is None:
            from reportlab.rl_config import invariant
        t = os.environ.get('SOURCE_DATE_EPOCH', '').strip()
        if invariant or t:
            t = int(t) if t else 946684800.0
            lt = time.gmtime(t)
            dhh = dmm = 0
            self.tzname = 'UTC'
        else:
            t = time.time()
            lt = tuple(time.localtime(t))
            dhh = int(time.timezone / 3600.0)
            dmm = time.timezone % 3600 % 60
            self.tzname = ''
        self.t = t
        self.lt = lt
        self.YMDhms = tuple(lt)[:6]
        self.dhh = dhh
        self.dmm = dmm

    @property
    def datetime(self):
        if self.tzname:
            return datetime.datetime.fromtimestamp(self.t, FixedOffsetTZ(self.dhh, self.dmm, self.tzname))
        else:
            return datetime.datetime.now()

    @property
    def asctime(self):
        return time.asctime(self.lt)