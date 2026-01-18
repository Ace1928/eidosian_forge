import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
class VersionError(Exception):
    """Indicates that the version does not conform to the required format"""
    is_user_error = True

    def __init__(self, version):
        self._version = version
        super(VersionError, self).__init__()

    def __str__(self):
        return 'Could not parse version: ' + self._version