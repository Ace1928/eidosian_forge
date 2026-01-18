import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def _pytest_has_xdist():
    """
    Check if the pytest-xdist plugin is installed, providing parallel tests
    """
    from importlib.util import find_spec
    return find_spec('xdist') is not None