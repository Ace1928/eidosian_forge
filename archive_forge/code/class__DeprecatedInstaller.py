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
class _DeprecatedInstaller(SetuptoolsDeprecationWarning):
    _SUMMARY = 'setuptools.installer and fetch_build_eggs are deprecated.'
    _DETAILS = '\n    Requirements should be satisfied by a PEP 517 installer.\n    If you are using pip, you can try `pip install --use-pep517`.\n    '