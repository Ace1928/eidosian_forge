import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
def _open_if_needed(self):
    if self._fileobj is None:
        self._fileobj = self._open_function(self._filename, self._mode)
        if self._start_byte != 0:
            self._fileobj.seek(self._start_byte)