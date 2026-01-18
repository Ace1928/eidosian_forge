from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGpmMetricsGet_t(_PrintableStructure):
    _fields_ = [('version', c_uint), ('numMetrics', c_uint), ('sample1', c_nvmlGpmSample_t), ('sample2', c_nvmlGpmSample_t), ('metrics', c_nvmlGpmMetric_t * NVML_GPM_METRIC_MAX)]