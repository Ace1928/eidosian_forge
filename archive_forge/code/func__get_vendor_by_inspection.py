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
def _get_vendor_by_inspection(self):
    """Return the vendor or None by checking for known SSH implementations."""
    version = self._get_ssh_version_string(['ssh', '-V'])
    return self._get_vendor_by_version_string(version, 'ssh')