from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlGridLicensableFeatures_v2_t(_PrintableStructure):
    _fields_ = [('isGridLicenseSupported', c_int), ('licensableFeaturesCount', c_uint), ('gridLicensableFeatures', c_nvmlGridLicensableFeature_v2_t * NVML_GRID_LICENSE_FEATURE_MAX_COUNT)]