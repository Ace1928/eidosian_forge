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
def TAG_ATTRIBUTES():
    return {'ImageWidth': 'imagewidth', 'ImageLength': 'imagelength', 'BitsPerSample': 'bitspersample', 'Compression': 'compression', 'PlanarConfiguration': 'planarconfig', 'FillOrder': 'fillorder', 'PhotometricInterpretation': 'photometric', 'ColorMap': 'colormap', 'ImageDescription': 'description', 'ImageDescription1': 'description1', 'SamplesPerPixel': 'samplesperpixel', 'RowsPerStrip': 'rowsperstrip', 'Software': 'software', 'Predictor': 'predictor', 'TileWidth': 'tilewidth', 'TileLength': 'tilelength', 'ExtraSamples': 'extrasamples', 'SampleFormat': 'sampleformat', 'ImageDepth': 'imagedepth', 'TileDepth': 'tiledepth'}