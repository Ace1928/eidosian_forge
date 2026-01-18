import glob
import os
import subprocess
import sys
import tempfile
from distutils import log
from distutils.errors import DistutilsError
from functools import partial
from . import _reqs
from .wheel import Wheel
from .warnings import SetuptoolsDeprecationWarning
def _fixup_find_links(find_links):
    """Ensure find-links option end-up being a list of strings."""
    if isinstance(find_links, str):
        return find_links.split()
    assert isinstance(find_links, (tuple, list))
    return find_links