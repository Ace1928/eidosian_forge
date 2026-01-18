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
class AbsoluteLinkError(FilterError):

    def __init__(self, tarinfo):
        self.tarinfo = tarinfo
        super().__init__(f'{tarinfo.name!r} is a link to an absolute path')