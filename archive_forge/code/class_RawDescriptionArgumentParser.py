import argparse
import logging
import os
import stat
import threading
import time
from errno import EIO, ENOENT
from fuse import FUSE, FuseOSError, LoggingMixIn, Operations
from fsspec import __version__
from fsspec.core import url_to_fs
class RawDescriptionArgumentParser(argparse.ArgumentParser):

    def format_help(self):
        usage = super().format_help()
        parts = usage.split('\n\n')
        parts[1] = self.description.rstrip()
        return '\n\n'.join(parts)