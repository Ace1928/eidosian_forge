from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
def _load_fontmanager(*, try_read_cache=True):
    fm_path = Path(mpl.get_cachedir(), f'fontlist-v{FontManager.__version__}.json')
    if try_read_cache:
        try:
            fm = json_load(fm_path)
        except Exception:
            pass
        else:
            if getattr(fm, '_version', object()) == FontManager.__version__:
                _log.debug('Using fontManager instance from %s', fm_path)
                return fm
    fm = FontManager()
    json_dump(fm, fm_path)
    _log.info('generated new fontManager')
    return fm