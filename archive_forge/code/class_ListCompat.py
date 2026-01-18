import os
import warnings
from ..core import Command
from ..errors import DistutilsPlatformError, DistutilsOptionError
from ..util import get_platform
class ListCompat(dict):

    def append(self, item):
        warnings.warn('format_commands is now a dict. append is deprecated.', DeprecationWarning, stacklevel=2)