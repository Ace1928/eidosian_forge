from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __fromDeviceQueryString(queryString):
    parameters = queryString.split(',')
    values = []
    for p in parameters:
        ps = p.strip()
        if ps in NVSMI_QUERY_GPU:
            values.append(NVSMI_QUERY_GPU[ps])
    return values