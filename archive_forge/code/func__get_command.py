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
def _get_command(self, old_path, new_path):
    my_map = {'old_path': old_path, 'new_path': new_path}
    command = [t.format(**my_map) for t in self.command_template]
    if command == self.command_template:
        command += [old_path, new_path]
    if sys.platform == 'win32':
        command_encoded = []
        for c in command:
            if isinstance(c, str):
                command_encoded.append(c.encode('mbcs'))
            else:
                command_encoded.append(c)
        return command_encoded
    else:
        return command