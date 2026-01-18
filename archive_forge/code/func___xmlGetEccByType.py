from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __xmlGetEccByType(handle, counterType, errorType):
    strResult = ''
    try:
        deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_DEVICE_MEMORY)
    except NVMLError as err:
        deviceMemory = nvidia_smi.__handleError(err)
    strResult += '          <device_memory>' + nvidia_smi.__toString(deviceMemory) + '</device_memory>\n'
    try:
        registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_DRAM)
    except NVMLError as err:
        registerFile = nvidia_smi.__handleError(err)
    strResult += '          <dram>' + nvidia_smi.__toString(registerFile) + '</dram>\n'
    try:
        registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_REGISTER_FILE)
    except NVMLError as err:
        registerFile = nvidia_smi.__handleError(err)
    strResult += '          <register_file>' + nvidia_smi.__toString(registerFile) + '</register_file>\n'
    try:
        l1Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_L1_CACHE)
    except NVMLError as err:
        l1Cache = nvidia_smi.__handleError(err)
    strResult += '          <l1_cache>' + nvidia_smi.__toString(l1Cache) + '</l1_cache>\n'
    try:
        l2Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_L2_CACHE)
    except NVMLError as err:
        l2Cache = nvidia_smi.__handleError(err)
    strResult += '          <l2_cache>' + nvidia_smi.__toString(l2Cache) + '</l2_cache>\n'
    try:
        textureMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_TEXTURE_MEMORY)
    except NVMLError as err:
        textureMemory = nvidia_smi.__handleError(err)
    strResult += '          <texture_memory>' + nvidia_smi.__toString(textureMemory) + '</texture_memory>\n'
    try:
        registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_CBU)
    except NVMLError as err:
        registerFile = nvidia_smi.__handleError(err)
    strResult += '          <cbu>' + nvidia_smi.__toString(registerFile) + '</cbu>\n'
    try:
        registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_SRAM)
    except NVMLError as err:
        registerFile = nvidia_smi.__handleError(err)
    strResult += '          <sram>' + nvidia_smi.__toString(registerFile) + '</sram>\n'
    try:
        count = nvidia_smi.__toString(nvmlDeviceGetTotalEccErrors(handle, errorType, counterType))
    except NVMLError as err:
        count = nvidia_smi.__handleError(err)
    strResult += '          <total>' + count + '</total>\n'
    return strResult