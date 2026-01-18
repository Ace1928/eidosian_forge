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
def _warn_wheel_not_available(dist):
    import pkg_resources
    try:
        pkg_resources.get_distribution('wheel')
    except pkg_resources.DistributionNotFound:
        dist.announce('WARNING: The wheel package is not available.', log.WARN)