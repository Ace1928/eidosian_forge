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
def NIH_CURVEFIT_TYPE():
    return ('StraightLine', 'Poly2', 'Poly3', 'Poly4', 'Poly5', 'ExpoFit', 'PowerFit', 'LogFit', 'RodbardFit', 'SpareFit1', 'Uncalibrated', 'UncalibratedOD')