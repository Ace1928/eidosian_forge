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
def COMPRESSION():

    class COMPRESSION(enum.IntEnum):
        NONE = 1
        CCITTRLE = 2
        CCITT_T4 = 3
        CCITT_T6 = 4
        LZW = 5
        OJPEG = 6
        JPEG = 7
        ADOBE_DEFLATE = 8
        JBIG_BW = 9
        JBIG_COLOR = 10
        JPEG_99 = 99
        KODAK_262 = 262
        NEXT = 32766
        SONY_ARW = 32767
        PACKED_RAW = 32769
        SAMSUNG_SRW = 32770
        CCIRLEW = 32771
        SAMSUNG_SRW2 = 32772
        PACKBITS = 32773
        THUNDERSCAN = 32809
        IT8CTPAD = 32895
        IT8LW = 32896
        IT8MP = 32897
        IT8BL = 32898
        PIXARFILM = 32908
        PIXARLOG = 32909
        DEFLATE = 32946
        DCS = 32947
        APERIO_JP2000_YCBC = 33003
        APERIO_JP2000_RGB = 33005
        JBIG = 34661
        SGILOG = 34676
        SGILOG24 = 34677
        JPEG2000 = 34712
        NIKON_NEF = 34713
        JBIG2 = 34715
        MDI_BINARY = 34718
        MDI_PROGRESSIVE = 34719
        MDI_VECTOR = 34720
        JPEG_LOSSY = 34892
        LZMA = 34925
        ZSTD = 34926
        OPS_PNG = 34933
        OPS_JPEGXR = 34934
        PIXTIFF = 50013
        KODAK_DCR = 65000
        PENTAX_PEF = 65535
    return COMPRESSION