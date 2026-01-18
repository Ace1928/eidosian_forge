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
def JM_fitz_config():
    have_TOFU = not hasattr(mupdf, 'TOFU')
    have_TOFU_BASE14 = not hasattr(mupdf, 'TOFU_BASE14')
    have_TOFU_CJK = not hasattr(mupdf, 'TOFU_CJK')
    have_TOFU_CJK_EXT = not hasattr(mupdf, 'TOFU_CJK_EXT')
    have_TOFU_CJK_LANG = not hasattr(mupdf, 'TOFU_CJK_LANG')
    have_TOFU_EMOJI = not hasattr(mupdf, 'TOFU_EMOJI')
    have_TOFU_HISTORIC = not hasattr(mupdf, 'TOFU_HISTORIC')
    have_TOFU_SIL = not hasattr(mupdf, 'TOFU_SIL')
    have_TOFU_SYMBOL = not hasattr(mupdf, 'TOFU_SYMBOL')
    ret = dict()
    ret['base14'] = have_TOFU_BASE14
    ret['cbz'] = bool(mupdf.FZ_ENABLE_CBZ)
    ret['epub'] = bool(mupdf.FZ_ENABLE_EPUB)
    ret['html'] = bool(mupdf.FZ_ENABLE_HTML)
    ret['icc'] = bool(mupdf.FZ_ENABLE_ICC)
    ret['img'] = bool(mupdf.FZ_ENABLE_IMG)
    ret['jpx'] = bool(mupdf.FZ_ENABLE_JPX)
    ret['js'] = bool(mupdf.FZ_ENABLE_JS)
    ret['pdf'] = bool(mupdf.FZ_ENABLE_PDF)
    ret['plotter-cmyk'] = bool(mupdf.FZ_PLOTTERS_CMYK)
    ret['plotter-g'] = bool(mupdf.FZ_PLOTTERS_G)
    ret['plotter-n'] = bool(mupdf.FZ_PLOTTERS_N)
    ret['plotter-rgb'] = bool(mupdf.FZ_PLOTTERS_RGB)
    ret['py-memory'] = bool(JM_MEMORY)
    ret['svg'] = bool(mupdf.FZ_ENABLE_SVG)
    ret['tofu'] = have_TOFU
    ret['tofu-cjk'] = have_TOFU_CJK
    ret['tofu-cjk-ext'] = have_TOFU_CJK_EXT
    ret['tofu-cjk-lang'] = have_TOFU_CJK_LANG
    ret['tofu-emoji'] = have_TOFU_EMOJI
    ret['tofu-historic'] = have_TOFU_HISTORIC
    ret['tofu-sil'] = have_TOFU_SIL
    ret['tofu-symbol'] = have_TOFU_SYMBOL
    ret['xps'] = bool(mupdf.FZ_ENABLE_XPS)
    return ret