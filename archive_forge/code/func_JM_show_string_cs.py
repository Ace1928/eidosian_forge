import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def JM_show_string_cs(text, user_font, trm, s, wmode, bidi_level, markup_dir, language):
    i = 0
    while i < len(s):
        l, ucs = mupdf.fz_chartorune(s[i:])
        i += l
        gid = mupdf.fz_encode_character_sc(user_font, ucs)
        if gid == 0:
            gid, font = mupdf.fz_encode_character_with_fallback(user_font, ucs, 0, language)
        else:
            font = user_font
        mupdf.fz_show_glyph(text, font, trm, gid, ucs, wmode, bidi_level, markup_dir, language)
        adv = mupdf.fz_advance_glyph(font, gid, wmode)
        if wmode == 0:
            trm = mupdf.fz_pre_translate(trm, adv, 0)
        else:
            trm = mupdf.fz_pre_translate(trm, 0, -adv)
    return trm