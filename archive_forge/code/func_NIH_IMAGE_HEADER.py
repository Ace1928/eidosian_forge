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
def NIH_IMAGE_HEADER():
    return [('FileID', 'a8'), ('nLines', 'i2'), ('PixelsPerLine', 'i2'), ('Version', 'i2'), ('OldLutMode', 'i2'), ('OldnColors', 'i2'), ('Colors', 'u1', (3, 32)), ('OldColorStart', 'i2'), ('ColorWidth', 'i2'), ('ExtraColors', 'u2', (6, 3)), ('nExtraColors', 'i2'), ('ForegroundIndex', 'i2'), ('BackgroundIndex', 'i2'), ('XScale', 'f8'), ('Unused2', 'i2'), ('Unused3', 'i2'), ('UnitsID', 'i2'), ('p1', [('x', 'i2'), ('y', 'i2')]), ('p2', [('x', 'i2'), ('y', 'i2')]), ('CurveFitType', 'i2'), ('nCoefficients', 'i2'), ('Coeff', 'f8', 6), ('UMsize', 'u1'), ('UM', 'a15'), ('UnusedBoolean', 'u1'), ('BinaryPic', 'b1'), ('SliceStart', 'i2'), ('SliceEnd', 'i2'), ('ScaleMagnification', 'f4'), ('nSlices', 'i2'), ('SliceSpacing', 'f4'), ('CurrentSlice', 'i2'), ('FrameInterval', 'f4'), ('PixelAspectRatio', 'f4'), ('ColorStart', 'i2'), ('ColorEnd', 'i2'), ('nColors', 'i2'), ('Fill1', '3u2'), ('Fill2', '3u2'), ('Table', 'u1'), ('LutMode', 'u1'), ('InvertedTable', 'b1'), ('ZeroClip', 'b1'), ('XUnitSize', 'u1'), ('XUnit', 'a11'), ('StackType', 'i2')]