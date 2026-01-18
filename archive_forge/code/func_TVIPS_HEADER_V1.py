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
def TVIPS_HEADER_V1():
    return [('Version', 'i4'), ('CommentV1', 'a80'), ('HighTension', 'i4'), ('SphericalAberration', 'i4'), ('IlluminationAperture', 'i4'), ('Magnification', 'i4'), ('PostMagnification', 'i4'), ('FocalLength', 'i4'), ('Defocus', 'i4'), ('Astigmatism', 'i4'), ('AstigmatismDirection', 'i4'), ('BiprismVoltage', 'i4'), ('SpecimenTiltAngle', 'i4'), ('SpecimenTiltDirection', 'i4'), ('IlluminationTiltDirection', 'i4'), ('IlluminationTiltAngle', 'i4'), ('ImageMode', 'i4'), ('EnergySpread', 'i4'), ('ChromaticAberration', 'i4'), ('ShutterType', 'i4'), ('DefocusSpread', 'i4'), ('CcdNumber', 'i4'), ('CcdSize', 'i4'), ('OffsetXV1', 'i4'), ('OffsetYV1', 'i4'), ('PhysicalPixelSize', 'i4'), ('Binning', 'i4'), ('ReadoutSpeed', 'i4'), ('GainV1', 'i4'), ('SensitivityV1', 'i4'), ('ExposureTimeV1', 'i4'), ('FlatCorrected', 'i4'), ('DeadPxCorrected', 'i4'), ('ImageMean', 'i4'), ('ImageStd', 'i4'), ('DisplacementX', 'i4'), ('DisplacementY', 'i4'), ('DateV1', 'i4'), ('TimeV1', 'i4'), ('ImageMin', 'i4'), ('ImageMax', 'i4'), ('ImageStatisticsQuality', 'i4')]