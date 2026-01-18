from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __handleError(err):
    if err.value == NVML_ERROR_NOT_SUPPORTED:
        return 'N/A'
    else:
        return err.__str__()