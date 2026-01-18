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
def _load_patterns_from_file(self, filename):
    lst = []
    with open(filename, 'r') as file_obj:
        for line in file_obj.readlines():
            pattern = line.strip()
            if not pattern or pattern.startswith('#'):
                continue
            lst.append(pattern)
    return lst