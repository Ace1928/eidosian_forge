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
def TVIPS_HEADER_V2():
    return [('ImageName', 'V160'), ('ImageFolder', 'V160'), ('ImageSizeX', 'i4'), ('ImageSizeY', 'i4'), ('ImageSizeZ', 'i4'), ('ImageSizeE', 'i4'), ('ImageDataType', 'i4'), ('Date', 'i4'), ('Time', 'i4'), ('Comment', 'V1024'), ('ImageHistory', 'V1024'), ('Scaling', '16f4'), ('ImageStatistics', '16c16'), ('ImageType', 'i4'), ('ImageDisplaType', 'i4'), ('PixelSizeX', 'f4'), ('PixelSizeY', 'f4'), ('ImageDistanceZ', 'f4'), ('ImageDistanceE', 'f4'), ('ImageMisc', '32f4'), ('TemType', 'V160'), ('TemHighTension', 'f4'), ('TemAberrations', '32f4'), ('TemEnergy', '32f4'), ('TemMode', 'i4'), ('TemMagnification', 'f4'), ('TemMagnificationCorrection', 'f4'), ('PostMagnification', 'f4'), ('TemStageType', 'i4'), ('TemStagePosition', '5f4'), ('TemImageShift', '2f4'), ('TemBeamShift', '2f4'), ('TemBeamTilt', '2f4'), ('TilingParameters', '7f4'), ('TemIllumination', '3f4'), ('TemShutter', 'i4'), ('TemMisc', '32f4'), ('CameraType', 'V160'), ('PhysicalPixelSizeX', 'f4'), ('PhysicalPixelSizeY', 'f4'), ('OffsetX', 'i4'), ('OffsetY', 'i4'), ('BinningX', 'i4'), ('BinningY', 'i4'), ('ExposureTime', 'f4'), ('Gain', 'f4'), ('ReadoutRate', 'f4'), ('FlatfieldDescription', 'V160'), ('Sensitivity', 'f4'), ('Dose', 'f4'), ('CamMisc', '32f4'), ('FeiMicroscopeInformation', 'V1024'), ('FeiSpecimenInformation', 'V1024'), ('Magic', 'u4')]