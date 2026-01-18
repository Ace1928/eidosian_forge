import sys
import re
import os
from configparser import RawConfigParser
class PkgNotFound(OSError):
    """Exception raised when a package can not be located."""

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg