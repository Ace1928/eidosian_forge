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
@property
def FILE_PATTERNS(self) -> dict[str, str]:
    return {'axes': '(?ix)\n                # matches Olympus OIF and Leica TIFF series\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                '}