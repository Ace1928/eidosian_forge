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
def _lsm_load_pages(self):
    """Load all pages from LSM file."""
    self.pages.cache = True
    self.pages.useframes = True
    self.pages.keyframe = 1
    keyframe = self.pages[1]
    for page in self.pages[1::2]:
        page.keyframe = keyframe
    self.pages.keyframe = 0
    keyframe = self.pages[0]
    for page in self.pages[::2]:
        page.keyframe = keyframe