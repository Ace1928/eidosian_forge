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
def __glob(self, path, do_glob):
    if do_glob:
        return glob.iglob(path)
    else:
        return [path]