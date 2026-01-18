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
@cached_property
def AXES_CODES(self) -> dict[str, str]:
    """Map dimension names to axes character codes.

        Reverse mapping of :py:attr:`AXES_NAMES`.

        """
    codes = {name: code for code, name in TIFF.AXES_NAMES.items()}
    codes['z'] = 'Z'
    codes['position'] = 'R'
    return codes