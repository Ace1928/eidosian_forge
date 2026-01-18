import os
import re
import unicodedata as ud
from .. import osutils, tests
from .._termcolor import FG, color_string
from ..tests.features import UnicodeFilenameFeature
def _mk_versioned_dir(self, path):
    self._mk_dir(path, versioned=True)