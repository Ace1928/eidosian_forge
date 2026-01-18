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
@staticmethod
def _check_hostname(arg):
    if arg.startswith('-'):
        raise StrangeHostname(hostname=arg)