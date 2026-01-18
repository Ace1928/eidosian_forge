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
def _fix_lsm_bitspersample(self, parent):
    """Correct LSM bitspersample tag.

        Old LSM writers may use a separate region for two 16-bit values,
        although they fit into the tag value element of the tag.

        """
    if self.code == 258 and self.count == 2:
        warnings.warn('correcting LSM bitspersample tag')
        tof = parent.offsetformat[parent.offsetsize]
        self.valueoffset = struct.unpack(tof, self._value)[0]
        parent.filehandle.seek(self.valueoffset)
        self.value = struct.unpack('<HH', parent.filehandle.read(4))