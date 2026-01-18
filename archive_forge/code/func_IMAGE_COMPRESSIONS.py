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
def IMAGE_COMPRESSIONS(self) -> set[int]:
    return {6, 7, 22610, 33003, 33004, 33005, 33007, 34712, 34892, 34933, 34934, 48124, 50001, 50002}