from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
class NullContext(object):
    """Null context manager.

    >>> with NullContext():
    ...     pass

    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass