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
def ExecCopyBundleResource(self, source, dest, convert_to_binary):
    """Copies a resource file to the bundle/Resources directory, performing any
    necessary compilation on each resource."""
    convert_to_binary = convert_to_binary == 'True'
    extension = os.path.splitext(source)[1].lower()
    if os.path.isdir(source):
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
    elif extension == '.xib':
        return self._CopyXIBFile(source, dest)
    elif extension == '.storyboard':
        return self._CopyXIBFile(source, dest)
    elif extension == '.strings' and (not convert_to_binary):
        self._CopyStringsFile(source, dest)
    else:
        if os.path.exists(dest):
            os.unlink(dest)
        shutil.copy(source, dest)
    if convert_to_binary and extension in ('.plist', '.strings'):
        self._ConvertToBinary(dest)