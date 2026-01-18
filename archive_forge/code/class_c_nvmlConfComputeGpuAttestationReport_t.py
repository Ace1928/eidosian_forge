from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
class c_nvmlConfComputeGpuAttestationReport_t(Structure):
    _fields_ = [('isCecAttestationReportPresent', c_uint), ('attestationReportSize', c_uint), ('cecAttestationReportSize', c_uint), ('nonce', c_uint8 * NVML_CC_GPU_CEC_NONCE_SIZE), ('attestationReport', c_uint8 * NVML_CC_GPU_ATTESTATION_REPORT_SIZE), ('cecAttestationReport', c_uint8 * NVML_CC_GPU_CEC_ATTESTATION_REPORT_SIZE)]