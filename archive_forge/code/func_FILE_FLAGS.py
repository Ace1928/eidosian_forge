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
def FILE_FLAGS():
    exclude = set('reduced final memmappable contiguous tiled chroma_subsampled'.split())
    return set((a[3:] for a in dir(TiffPage) if a[:3] == 'is_' and a[3:] not in exclude))