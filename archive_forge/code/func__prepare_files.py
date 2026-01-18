import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
def _prepare_files(self, old_path, new_path, force_temp=False, allow_write_new=False):
    old_disk_path = self._write_file(old_path, self.old_tree, 'old', force_temp)
    new_disk_path = self._write_file(new_path, self.new_tree, 'new', force_temp, allow_write=allow_write_new)
    return (old_disk_path, new_disk_path)