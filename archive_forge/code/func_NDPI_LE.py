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
def NDPI_LE(self) -> TiffFormat:
    """32-bit little-endian TIFF format with 64-bit offsets."""
    return TiffFormat(version=42, byteorder='<', offsetsize=8, offsetformat='<Q', tagnosize=2, tagnoformat='<H', tagsize=12, tagformat1='<HH', tagformat2='<I8s', tagoffsetthreshold=4)