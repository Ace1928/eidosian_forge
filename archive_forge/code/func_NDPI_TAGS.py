from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@cached_property
def NDPI_TAGS(self) -> TiffTagRegistry:
    """Registry of private TIFF tags for Hamamatsu NDPI (65420-65458)."""
    return TiffTagRegistry(((65324, 'OffsetHighBytes'), (65325, 'ByteCountHighBytes'), (65420, 'FileFormat'), (65421, 'Magnification'), (65422, 'XOffsetFromSlideCenter'), (65423, 'YOffsetFromSlideCenter'), (65424, 'ZOffsetFromSlideCenter'), (65425, 'TissueIndex'), (65426, 'McuStarts'), (65427, 'SlideLabel'), (65428, 'AuthCode'), (65429, '65429'), (65430, '65430'), (65431, '65431'), (65432, 'McuStartsHighBytes'), (65433, '65433'), (65434, 'Fluorescence'), (65435, 'ExposureRatio'), (65436, 'RedMultiplier'), (65437, 'GreenMultiplier'), (65438, 'BlueMultiplier'), (65439, 'FocusPoints'), (65440, 'FocusPointRegions'), (65441, 'CaptureMode'), (65442, 'ScannerSerialNumber'), (65443, '65443'), (65444, 'JpegQuality'), (65445, 'RefocusInterval'), (65446, 'FocusOffset'), (65447, 'BlankLines'), (65448, 'FirmwareVersion'), (65449, 'Comments'), (65450, 'LabelObscured'), (65451, 'Wavelength'), (65452, '65452'), (65453, 'LampAge'), (65454, 'ExposureTime'), (65455, 'FocusTime'), (65456, 'ScanTime'), (65457, 'WriteTime'), (65458, 'FullyAutoFocus'), (65500, 'DefaultGamma')))