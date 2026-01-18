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
class FunctionContainer(object):
    """An object that contains a function and any args or kwargs to call it

    When called the provided function will be called with provided args
    and kwargs.
    """

    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __repr__(self):
        return 'Function: %s with args %s and kwargs %s' % (self._func, self._args, self._kwargs)

    def __call__(self):
        return self._func(*self._args, **self._kwargs)