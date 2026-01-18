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
def _LoadPlistMaybeBinary(self, plist_path):
    """Loads into a memory a plist possibly encoded in binary format.

    This is a wrapper around plistlib.readPlist that tries to convert the
    plist to the XML format if it can't be parsed (assuming that it is in
    the binary format).

    Args:
      plist_path: string, path to a plist file, in XML or binary format

    Returns:
      Content of the plist as a dictionary.
    """
    try:
        return plistlib.readPlist(plist_path)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile() as temp:
        shutil.copy2(plist_path, temp.name)
        subprocess.check_call(['plutil', '-convert', 'xml1', temp.name])
        return plistlib.readPlist(temp.name)