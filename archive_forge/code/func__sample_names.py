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
def _sample_names(self) -> list[str] | None:
    """Return names of samples."""
    if 'S' not in self.axes:
        return None
    samples = self.shape[self.axes.find('S')]
    extrasamples = len(self.extrasamples)
    if samples < 1 or extrasamples > 2:
        return None
    if self.photometric == 0:
        names = ['WhiteIsZero']
    elif self.photometric == 1:
        names = ['BlackIsZero']
    elif self.photometric == 2:
        names = ['Red', 'Green', 'Blue']
    elif self.photometric == 5:
        names = ['Cyan', 'Magenta', 'Yellow', 'Black']
    elif self.photometric == 6:
        if self.compression in {6, 7, 34892, 33007}:
            names = ['Red', 'Green', 'Blue']
        else:
            names = ['Luma', 'Cb', 'Cr']
    else:
        return None
    if extrasamples > 0:
        names += [enumarg(EXTRASAMPLE, self.extrasamples[0]).name.title()]
    if extrasamples > 1:
        names += [enumarg(EXTRASAMPLE, self.extrasamples[1]).name.title()]
    if len(names) != samples:
        return None
    return names