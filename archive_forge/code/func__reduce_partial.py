from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def _reduce_partial(p):
    return (_rebuild_partial, (p.func, p.args, p.keywords or {}))