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
def UIC_TAGS():
    from fractions import Fraction
    return [('AutoScale', int), ('MinScale', int), ('MaxScale', int), ('SpatialCalibration', int), ('XCalibration', Fraction), ('YCalibration', Fraction), ('CalibrationUnits', str), ('Name', str), ('ThreshState', int), ('ThreshStateRed', int), ('tagid_10', None), ('ThreshStateGreen', int), ('ThreshStateBlue', int), ('ThreshStateLo', int), ('ThreshStateHi', int), ('Zoom', int), ('CreateTime', julian_datetime), ('LastSavedTime', julian_datetime), ('currentBuffer', int), ('grayFit', None), ('grayPointCount', None), ('grayX', Fraction), ('grayY', Fraction), ('grayMin', Fraction), ('grayMax', Fraction), ('grayUnitName', str), ('StandardLUT', int), ('wavelength', int), ('StagePosition', '(%i,2,2)u4'), ('CameraChipOffset', '(%i,2,2)u4'), ('OverlayMask', None), ('OverlayCompress', None), ('Overlay', None), ('SpecialOverlayMask', None), ('SpecialOverlayCompress', None), ('SpecialOverlay', None), ('ImageProperty', read_uic_image_property), ('StageLabel', '%ip'), ('AutoScaleLoInfo', Fraction), ('AutoScaleHiInfo', Fraction), ('AbsoluteZ', '(%i,2)u4'), ('AbsoluteZValid', '(%i,)u4'), ('Gamma', 'I'), ('GammaRed', 'I'), ('GammaGreen', 'I'), ('GammaBlue', 'I'), ('CameraBin', '2I'), ('NewLUT', int), ('ImagePropertyEx', None), ('PlaneProperty', int), ('UserLutTable', '(256,3)u1'), ('RedAutoScaleInfo', int), ('RedAutoScaleLoInfo', Fraction), ('RedAutoScaleHiInfo', Fraction), ('RedMinScaleInfo', int), ('RedMaxScaleInfo', int), ('GreenAutoScaleInfo', int), ('GreenAutoScaleLoInfo', Fraction), ('GreenAutoScaleHiInfo', Fraction), ('GreenMinScaleInfo', int), ('GreenMaxScaleInfo', int), ('BlueAutoScaleInfo', int), ('BlueAutoScaleLoInfo', Fraction), ('BlueAutoScaleHiInfo', Fraction), ('BlueMinScaleInfo', int), ('BlueMaxScaleInfo', int)]