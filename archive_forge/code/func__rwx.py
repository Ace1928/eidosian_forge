import stat
import time
from paramiko.common import x80000000, o700, o70, xffffffff
@staticmethod
def _rwx(n, suid, sticky=False):
    if suid:
        suid = 2
    out = '-r'[n >> 2] + '-w'[n >> 1 & 1]
    if sticky:
        out += '-xTt'[suid + (n & 1)]
    else:
        out += '-xSs'[suid + (n & 1)]
    return out