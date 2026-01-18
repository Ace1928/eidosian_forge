import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def _set_version(self, version):
    if version is not None:
        self._raw_version = str(version)
    else:
        self._raw_version = None