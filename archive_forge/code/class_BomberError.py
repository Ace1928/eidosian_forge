import configparser
import glob
import os
import sys
from os.path import join as pjoin
from packaging.version import Version
from .environment import get_nipy_system_dir, get_nipy_user_dir
class BomberError(DataError, AttributeError):
    """Error when trying to access Bomber instance

    Should be instance of AttributeError to allow Python 3 inspect to do
    various ``hasattr`` checks without raising an error
    """
    pass