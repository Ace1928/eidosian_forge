import operator
import os
import re
import sys
def _get_versions(self, version):
    """Returns a tuple of major version, minor version"""
    return (version >> 16, version & 65535)