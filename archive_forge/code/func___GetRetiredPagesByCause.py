from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __GetRetiredPagesByCause(handle, cause):
    retiredPagedByCause = {}
    error = None
    count = 0
    try:
        pages = nvmlDeviceGetRetiredPages(handle, cause)
    except NVMLError as err:
        error = nvidia_smi.__handleError(err)
        pages = None
    retiredPageAddresses = {}
    if pages is not None:
        ii = 1
        for page in pages:
            retiredPageAddresses['retired_page_address_' + str(ii)] = '0x%016x' % page
            ii += 1
            count += 1
    if error is not None:
        retiredPageAddresses['Error'] = error
    retiredPagedByCause['retired_count'] = count
    retiredPagedByCause['retired_page_addresses'] = retiredPageAddresses if len(retiredPageAddresses.values()) > 0 else None
    return retiredPagedByCause if count > 0 else None