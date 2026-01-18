import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def Dispatch(self, args):
    """Dispatches a string command to a method."""
    if len(args) < 1:
        raise Exception('Not enough arguments')
    method = 'Exec%s' % self._CommandifyName(args[0])
    return getattr(self, method)(*args[1:])