import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def _get_vendor_from_path(self, path):
    """Return the vendor or None using the program at the given path"""
    version = self._get_ssh_version_string([path, '-V'])
    return self._get_vendor_by_version_string(version, os.path.splitext(os.path.basename(path))[0])