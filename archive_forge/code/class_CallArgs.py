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
class CallArgs(object):

    def __init__(self, **kwargs):
        """A class that records call arguments

        The call arguments must be passed as keyword arguments. It will set
        each keyword argument as an attribute of the object along with its
        associated value.
        """
        for arg, value in kwargs.items():
            setattr(self, arg, value)