import os
import re
import shutil
import subprocess
import stat
import string
import sys
def _GetEnv(self, arch):
    """Gets the saved environment from a file for a given architecture."""
    pairs = open(arch).read()[:-2].split('\x00')
    kvs = [item.split('=', 1) for item in pairs]
    return dict(kvs)