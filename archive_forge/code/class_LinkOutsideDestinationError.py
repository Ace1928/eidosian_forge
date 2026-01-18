from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
class LinkOutsideDestinationError(FilterError):

    def __init__(self, tarinfo, path):
        self.tarinfo = tarinfo
        self._path = path
        super().__init__(f'{tarinfo.name!r} would link to {path!r}, ' + 'which is outside the destination')