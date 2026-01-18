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
def _CopyStringsFile(self, source, dest):
    """Copies a .strings file using iconv to reconvert the input into UTF-16."""
    input_code = self._DetectInputEncoding(source) or 'UTF-8'
    import CoreFoundation
    with open(source, 'rb') as in_file:
        s = in_file.read()
    d = CoreFoundation.CFDataCreate(None, s, len(s))
    _, error = CoreFoundation.CFPropertyListCreateFromXMLData(None, d, 0, None)
    if error:
        return
    with open(dest, 'wb') as fp:
        fp.write(s.decode(input_code).encode('UTF-16'))