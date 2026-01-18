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
def _fontentry_helper_repr_png(fontent):
    from matplotlib.figure import Figure
    fig = Figure()
    font_path = Path(fontent.fname) if fontent.fname != '' else None
    fig.text(0, 0, fontent.name, font=font_path)
    with BytesIO() as buf:
        fig.savefig(buf, bbox_inches='tight', transparent=True)
        return buf.getvalue()