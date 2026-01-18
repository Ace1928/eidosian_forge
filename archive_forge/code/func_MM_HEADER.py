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
def MM_HEADER():
    MM_DIMENSION = [('Name', 'a16'), ('Size', 'i4'), ('Origin', 'f8'), ('Resolution', 'f8'), ('Unit', 'a64')]
    return [('HeaderFlag', 'i2'), ('ImageType', 'u1'), ('ImageName', 'a257'), ('OffsetData', 'u4'), ('PaletteSize', 'i4'), ('OffsetPalette0', 'u4'), ('OffsetPalette1', 'u4'), ('CommentSize', 'i4'), ('OffsetComment', 'u4'), ('Dimensions', MM_DIMENSION, 10), ('OffsetPosition', 'u4'), ('MapType', 'i2'), ('MapMin', 'f8'), ('MapMax', 'f8'), ('MinValue', 'f8'), ('MaxValue', 'f8'), ('OffsetMap', 'u4'), ('Gamma', 'f8'), ('Offset', 'f8'), ('GrayChannel', MM_DIMENSION), ('OffsetThumbnail', 'u4'), ('VoiceField', 'i4'), ('OffsetVoiceField', 'u4')]