from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlConfComputeGpuCertificate_t(Structure):
    _fields_ = [('certChainSize', c_uint), ('attestationCertChainSize', c_uint), ('certChain', c_uint8 * NVML_GPU_CERT_CHAIN_SIZE), ('attestationCertChain', c_uint8 * NVML_GPU_ATTESTATION_CERT_CHAIN_SIZE)]