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
def _lsm_fix_strip_bytecounts(self):
    """Set databytecounts to size of compressed data.

        The StripByteCounts tag in LSM files contains the number of bytes
        for the uncompressed data.

        """
    pages = self.pages
    if pages[0].compression == 1:
        return
    pages = sorted(pages, key=lambda p: p.dataoffsets[0])
    npages = len(pages) - 1
    for i, page in enumerate(pages):
        if page.index % 2:
            continue
        offsets = page.dataoffsets
        bytecounts = page.databytecounts
        if i < npages:
            lastoffset = pages[i + 1].dataoffsets[0]
        else:
            lastoffset = min(offsets[-1] + 2 * bytecounts[-1], self._fh.size)
        offsets = offsets + (lastoffset,)
        page.databytecounts = tuple((offsets[j + 1] - offsets[j] for j in range(len(bytecounts))))