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
def CZ_LSMINFO_SCANTYPE():
    return {0: 'XYZCT', 1: 'XYZCT', 2: 'XYZCT', 3: 'XYTCZ', 4: 'XYZTC', 5: 'XYTCZ', 6: 'XYZTC', 7: 'XYCTZ', 8: 'XYCZT', 9: 'XYTCZ', 10: 'XYZCT'}