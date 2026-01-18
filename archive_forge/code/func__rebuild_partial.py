from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def _rebuild_partial(func, args, keywords):
    return functools.partial(func, *args, **keywords)