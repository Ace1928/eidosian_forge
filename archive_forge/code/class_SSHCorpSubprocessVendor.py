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
class SSHCorpSubprocessVendor(SubprocessVendor):
    """SSH vendor that uses the 'ssh' executable from SSH Corporation."""
    executable_path = 'ssh'

    def _get_vendor_specific_argv(self, username, host, port, subsystem=None, command=None):
        self._check_hostname(host)
        args = [self.executable_path, '-x']
        if port is not None:
            args.extend(['-p', str(port)])
        if username is not None:
            args.extend(['-l', username])
        if subsystem is not None:
            args.extend(['-s', subsystem, host])
        else:
            args.extend([host] + command)
        return args