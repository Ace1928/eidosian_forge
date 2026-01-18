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
def AXES_LABELS():
    axes = {'X': 'width', 'Y': 'height', 'Z': 'depth', 'S': 'sample', 'I': 'series', 'T': 'time', 'C': 'channel', 'A': 'angle', 'P': 'phase', 'R': 'tile', 'H': 'lifetime', 'E': 'lambda', 'L': 'exposure', 'V': 'event', 'Q': 'other', 'M': 'mosaic'}
    axes.update(dict(((v, k) for k, v in axes.items())))
    return axes