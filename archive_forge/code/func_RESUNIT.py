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
def RESUNIT():

    class RESUNIT(enum.IntEnum):
        NONE = 1
        INCH = 2
        CENTIMETER = 3
    return RESUNIT