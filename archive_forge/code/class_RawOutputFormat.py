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
class RawOutputFormat:
    """
    Format string representations.
    """

    def __init__(self, format=None):
        self.format = format or {}

    def simple(self, s, attribute):
        if not isinstance(s, str):
            s = str(s)
        return self.format.get(attribute, '') + s + self.format.get('normal', '')

    def punctuation(self, s):
        """Punctuation color."""
        return self.simple(s, 'normal')

    def field_value(self, s):
        """Field value color."""
        return self.simple(s, 'purple')

    def field_name(self, s):
        """Field name color."""
        return self.simple(s, 'blue')

    def class_name(self, s):
        """Class name color."""
        return self.format.get('red', '') + self.simple(s, 'bold')