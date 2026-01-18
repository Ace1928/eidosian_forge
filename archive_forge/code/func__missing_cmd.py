from __future__ import annotations
from packaging.version import Version
import param
from .core import Canvas                                 # noqa (API import)
from .reductions import *                                # noqa (API import)
from .glyphs import Point                                # noqa (API import)
from .pipeline import Pipeline                           # noqa (API import)
from . import transfer_functions as tf                   # noqa (API import)
from . import data_libraries                             # noqa (API import)
from pandas import __version__ as pandas_version
from functools import partial
def _missing_cmd(*args, **kw):
    return 'install pyct to enable this command (e.g. `conda install pyct or `pip install pyct[cmd]`)'