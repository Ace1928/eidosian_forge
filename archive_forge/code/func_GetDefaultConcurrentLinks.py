import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def GetDefaultConcurrentLinks():
    """Returns a best-guess for a number of concurrent links."""
    pool_size = int(os.environ.get('GYP_LINK_CONCURRENCY', 0))
    if pool_size:
        return pool_size
    if sys.platform in ('win32', 'cygwin'):
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('sullAvailExtendedVirtual', ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        mem_limit = max(1, stat.ullTotalPhys // (5 * 2 ** 30))
        hard_cap = max(1, int(os.environ.get('GYP_LINK_CONCURRENCY_MAX', 2 ** 32)))
        return min(mem_limit, hard_cap)
    elif sys.platform.startswith('linux'):
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo') as meminfo:
                memtotal_re = re.compile('^MemTotal:\\s*(\\d*)\\s*kB')
                for line in meminfo:
                    match = memtotal_re.match(line)
                    if not match:
                        continue
                    return max(1, int(match.group(1)) // (8 * 2 ** 20))
        return 1
    elif sys.platform == 'darwin':
        try:
            avail_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']))
            return max(1, avail_bytes // (4 * 2 ** 30))
        except subprocess.CalledProcessError:
            return 1
    else:
        return 1