import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def _win32_fixdrive(path):
    """Force drive letters to be consistent.

    win32 is inconsistent whether it returns lower or upper case
    and even if it was consistent the user might type the other
    so we force it to uppercase
    running python.exe under cmd.exe return capital C:\\
    running win32 python inside a cygwin shell returns lowercase c:\\
    """
    drive, path = ntpath.splitdrive(path)
    return drive.upper() + path