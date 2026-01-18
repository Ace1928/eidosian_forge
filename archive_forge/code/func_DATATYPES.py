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
def DATATYPES():

    class DATATYPES(enum.IntEnum):
        NOTYPE = 0
        BYTE = 1
        ASCII = 2
        SHORT = 3
        LONG = 4
        RATIONAL = 5
        SBYTE = 6
        UNDEFINED = 7
        SSHORT = 8
        SLONG = 9
        SRATIONAL = 10
        FLOAT = 11
        DOUBLE = 12
        IFD = 13
        UNICODE = 14
        COMPLEX = 15
        LONG8 = 16
        SLONG8 = 17
        IFD8 = 18
    return DATATYPES