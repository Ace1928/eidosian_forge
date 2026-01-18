from __future__ import print_function
import errno
import logging
import os
import time
from oauth2client import util
class CredentialsFileSymbolicLinkError(Exception):
    """Credentials files must not be symbolic links."""