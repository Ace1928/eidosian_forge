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
def _CommandifyName(self, name_string):
    """Transforms a tool name like copy-info-plist to CopyInfoPlist"""
    return name_string.title().replace('-', '')