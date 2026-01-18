import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
def __walk_rec(self, top, rec):
    """
        Yields each subdirectories of top, doesn't follow symlinks.
        If rec is false, only yield top.

        @param top: root directory.
        @type top: string
        @param rec: recursive flag.
        @type rec: bool
        @return: path of one subdirectory.
        @rtype: string
        """
    if not rec or os.path.islink(top) or (not os.path.isdir(top)):
        yield top
    else:
        for root, dirs, files in os.walk(top):
            yield root