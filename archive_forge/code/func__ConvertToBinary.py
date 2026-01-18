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
def _ConvertToBinary(self, dest):
    subprocess.check_call(['xcrun', 'plutil', '-convert', 'binary1', '-o', dest, dest])