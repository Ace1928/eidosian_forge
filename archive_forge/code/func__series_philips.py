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
def _series_philips(self) -> list[TiffPageSeries] | None:
    """Return pyramidal image series in Philips DP file."""
    series = self._series_generic()
    if series is None:
        return None
    for s in series:
        s.kind = 'philips'
        if s.is_pyramidal:
            s.name = 'Baseline'
        elif s.keyframe.description.startswith('Macro'):
            s.name = 'Macro'
        elif s.keyframe.description.startswith('Label'):
            s.name = 'Label'
    self.is_uniform = False
    return series