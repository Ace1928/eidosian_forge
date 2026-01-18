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
def _get_vendor_by_config(self):
    vendor_name = config.GlobalStack().get('ssh')
    if vendor_name is not None:
        try:
            vendor = self._ssh_vendors[vendor_name]
        except KeyError:
            vendor = self._get_vendor_from_path(vendor_name)
            if vendor is None:
                raise errors.UnknownSSH(vendor_name)
            vendor.executable_path = vendor_name
        return vendor
    return None