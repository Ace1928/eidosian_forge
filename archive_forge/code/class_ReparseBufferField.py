import os
import pathlib
import platform
import stat
import sys
from logging import getLogger
from typing import Union
class ReparseBufferField(ctypes.Union):
    _fields_ = [('symlink', SymbolicLinkReparseBuffer), ('mount', MountReparseBuffer)]