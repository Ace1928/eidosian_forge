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
class SAMPLEFORMAT(enum.IntEnum):
    UINT = 1
    INT = 2
    IEEEFP = 3
    VOID = 4
    COMPLEXINT = 5
    COMPLEXIEEEFP = 6