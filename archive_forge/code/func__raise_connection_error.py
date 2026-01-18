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
def _raise_connection_error(self, host, port=None, orig_error=None, msg='Unable to connect to SSH host'):
    """Raise a SocketConnectionError with properly formatted host.

        This just unifies all the locations that try to raise ConnectionError,
        so that they format things properly.
        """
    raise errors.SocketConnectionError(host=host, port=port, msg=msg, orig_error=orig_error)