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
def TAG_LOAD(self) -> frozenset[int]:
    return frozenset((258, 270, 273, 277, 279, 282, 283, 305, 324, 325, 330, 338, 339, 347, 513, 514, 530, 33628, 42113, 50838, 50839))