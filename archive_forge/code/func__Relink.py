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
def _Relink(self, dest, link):
    """Creates a symlink to |dest| named |link|. If |link| already exists,
    it is overwritten."""
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(dest, link)