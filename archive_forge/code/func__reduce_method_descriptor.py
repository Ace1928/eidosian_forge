from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def _reduce_method_descriptor(m):
    return (getattr, (m.__objclass__, m.__name__))