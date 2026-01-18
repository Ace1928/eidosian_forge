from __future__ import annotations
import logging  # isort:skip
from os import listdir
from os.path import join
from packaging.version import Version as V
from bokeh import __version__
from bokeh.resources import get_sri_hashes_for_version
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import RELEASE_DETAIL
 Required Sphinx extension setup function. 