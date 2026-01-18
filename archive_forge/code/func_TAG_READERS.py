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
def TAG_READERS():
    return {320: read_colormap, 33723: read_bytes, 33628: read_uic1tag, 33629: read_uic2tag, 33630: read_uic3tag, 33631: read_uic4tag, 34118: read_cz_sem, 34361: read_mm_header, 34362: read_mm_stamp, 34363: read_numpy, 34386: read_numpy, 34412: read_cz_lsminfo, 34680: read_fei_metadata, 34682: read_fei_metadata, 37706: read_tvips_header, 37724: read_bytes, 33923: read_bytes, 43314: read_nih_image_header, 40100: read_bytes, 50288: read_bytes, 50296: read_bytes, 50839: read_bytes, 51123: read_json, 34665: read_exif_ifd, 34853: read_gps_ifd, 40965: read_interoperability_ifd}