import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
class SavedFlag(object):
    """Helper class for saving and restoring a flag value."""

    def __init__(self, flag):
        self.flag = flag
        self.value = flag.value
        self.present = flag.present

    def RestoreFlag(self):
        self.flag.value = self.value
        self.flag.present = self.present