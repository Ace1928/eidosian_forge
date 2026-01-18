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
def _fluoview_series(self):
    """Return image series in FluoView file."""
    self.pages.useframes = True
    self.pages.keyframe = 0
    self.pages.load()
    mm = self.fluoview_metadata
    mmhd = list(reversed(mm['Dimensions']))
    axes = ''.join((TIFF.MM_DIMENSIONS.get(i[0].upper(), 'Q') for i in mmhd if i[1] > 1))
    shape = tuple((int(i[1]) for i in mmhd if i[1] > 1))
    return [TiffPageSeries(self.pages, shape, self.pages[0].dtype, axes, name=mm['ImageName'], stype='FluoView')]