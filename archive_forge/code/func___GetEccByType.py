from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __GetEccByType(handle, counterType, errorType):
    strResult = ''
    eccByType = {}
    try:
        deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_DEVICE_MEMORY)
    except NVMLError as err:
        deviceMemory = nvidia_smi.__handleError(err)
    eccByType['device_memory'] = deviceMemory
    try:
        deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_DRAM)
    except NVMLError as err:
        deviceMemory = nvidia_smi.__handleError(err)
    eccByType['dram'] = deviceMemory
    try:
        registerFile = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_REGISTER_FILE)
    except NVMLError as err:
        registerFile = nvidia_smi.__handleError(err)
    eccByType['register_file'] = registerFile
    try:
        l1Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_L1_CACHE)
    except NVMLError as err:
        l1Cache = nvidia_smi.__handleError(err)
    eccByType['l1_cache'] = l1Cache
    try:
        l2Cache = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_L2_CACHE)
    except NVMLError as err:
        l2Cache = nvidia_smi.__handleError(err)
    eccByType['l2_cache'] = l2Cache
    try:
        textureMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_TEXTURE_MEMORY)
    except NVMLError as err:
        textureMemory = nvidia_smi.__handleError(err)
    eccByType['texture_memory'] = textureMemory
    try:
        deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_CBU)
    except NVMLError as err:
        deviceMemory = nvidia_smi.__handleError(err)
    eccByType['cbu'] = deviceMemory
    try:
        deviceMemory = nvmlDeviceGetMemoryErrorCounter(handle, errorType, counterType, NVML_MEMORY_LOCATION_SRAM)
    except NVMLError as err:
        deviceMemory = nvidia_smi.__handleError(err)
    eccByType['sram'] = deviceMemory
    try:
        count = nvidia_smi.__toString(nvmlDeviceGetTotalEccErrors(handle, errorType, counterType))
    except NVMLError as err:
        count = nvidia_smi.__handleError(err)
    eccByType['total'] = count
    return eccByType