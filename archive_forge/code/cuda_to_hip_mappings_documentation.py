import collections
import os
import re
import subprocess
from .constants import (API_BLAS, API_C10, API_CAFFE2, API_DRIVER, API_FFT,
 Mapping of CUDA functions, include files, constants, and types to ROCm/HIP equivalents
This closely follows the implementation in hipify-clang
https://github.com/ROCm-Developer-Tools/HIP/blob/master/hipify-clang/src/CUDA2HipMap.cpp
and its structure.
There are different maps for fundamental names, include files, identifies, sparse, and
PyTorch specific translations.
Each of the entries in these maps translates a CUDA string to a tuple containing the
ROCm/HIP string, a type and API annotation and - optionally - an annotation if it is not
supported in ROCm/HIP yet.
