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
class WatchManagerError(Exception):
    """
    WatchManager Exception. Raised on error encountered on watches
    operations.
    """

    def __init__(self, msg, wmd):
        """
        @param msg: Exception string's description.
        @type msg: string
        @param wmd: This dictionary contains the wd assigned to paths of the
                    same call for which watches were successfully added.
        @type wmd: dict
        """
        self.wmd = wmd
        Exception.__init__(self, msg)