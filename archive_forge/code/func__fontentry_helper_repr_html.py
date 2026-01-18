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
def _fontentry_helper_repr_html(fontent):
    png_stream = _fontentry_helper_repr_png(fontent)
    png_b64 = b64encode(png_stream).decode()
    return f'<img src="data:image/png;base64, {png_b64}" />'