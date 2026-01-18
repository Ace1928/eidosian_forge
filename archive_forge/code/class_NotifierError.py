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
class NotifierError(PyinotifyError):
    """
    Notifier Exception. Raised on Notifier error.

    """

    def __init__(self, err):
        """
        @param err: Exception string's description.
        @type err: string
        """
        PyinotifyError.__init__(self, err)