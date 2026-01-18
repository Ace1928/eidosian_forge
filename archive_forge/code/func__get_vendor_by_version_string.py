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
def _get_vendor_by_version_string(self, version, progname):
    """Return the vendor or None based on output from the subprocess.

        :param version: The output of 'ssh -V' like command.
        :param args: Command line that was run.
        """
    vendor = None
    if 'OpenSSH' in version:
        trace.mutter('ssh implementation is OpenSSH')
        vendor = OpenSSHSubprocessVendor()
    elif 'SSH Secure Shell' in version:
        trace.mutter('ssh implementation is SSH Corp.')
        vendor = SSHCorpSubprocessVendor()
    elif 'lsh' in version:
        trace.mutter('ssh implementation is GNU lsh.')
        vendor = LSHSubprocessVendor()
    elif 'plink' in version and progname == 'plink':
        trace.mutter("ssh implementation is Putty's plink.")
        vendor = PLinkSubprocessVendor()
    return vendor