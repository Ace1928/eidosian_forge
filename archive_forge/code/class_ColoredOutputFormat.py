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
class ColoredOutputFormat(RawOutputFormat):
    """
    Format colored string representations.
    """

    def __init__(self):
        f = {'normal': '\x1b[0m', 'black': '\x1b[30m', 'red': '\x1b[31m', 'green': '\x1b[32m', 'yellow': '\x1b[33m', 'blue': '\x1b[34m', 'purple': '\x1b[35m', 'cyan': '\x1b[36m', 'bold': '\x1b[1m', 'uline': '\x1b[4m', 'blink': '\x1b[5m', 'invert': '\x1b[7m'}
        RawOutputFormat.__init__(self, f)