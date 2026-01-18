from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def _ndpi_load_pages(self) -> None:
    """Read and fix pages from NDPI slide file if CaptureMode > 6.

        If the value of the CaptureMode tag is greater than 6, change the
        attributes of TiffPage instances that are part of the pyramid to
        match 16-bit grayscale data. TiffTag values are not corrected.

        """
    pages = self.pages
    capturemode = self.pages.first.tags.valueof(65441)
    if capturemode is None or capturemode < 6:
        return
    pages.cache = True
    pages.useframes = False
    pages._load()
    for page in pages:
        assert isinstance(page, TiffPage)
        mag = page.tags.valueof(65421)
        if mag is None or mag > 0:
            page.photometric = PHOTOMETRIC.MINISBLACK
            page.sampleformat = SAMPLEFORMAT.UINT
            page.samplesperpixel = 1
            page.bitspersample = 16
            page.dtype = page._dtype = numpy.dtype('uint16')
            if page.shaped[-1] > 1:
                page.axes = page.axes[:-1]
                page.shape = page.shape[:-1]
                page.shaped = page.shaped[:-1] + (1,)