import codecs
import io
import os
import sys
import warnings
from ..lazy_import import lazy_import
import time
from breezy import (
from .. import config, osutils, trace
from . import NullProgressView, UIFactory
def _repaint(self):
    s = self._render_line()
    self._show_line(s)
    self._have_output = True