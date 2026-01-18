import inspect
import sys
import os
import errno
import socket
from botocore.compat import six
from multiprocessing.managers import BaseManager
def accepts_kwargs(func):
    return inspect.getargspec(func)[2]