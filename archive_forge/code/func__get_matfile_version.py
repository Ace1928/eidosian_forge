import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
def _get_matfile_version(fileobj):
    fileobj.seek(0)
    mopt_bytes = fileobj.read(4)
    if len(mopt_bytes) == 0:
        raise MatReadError('Mat file appears to be empty')
    mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=mopt_bytes)
    if 0 in mopt_ints:
        fileobj.seek(0)
        return (0, 0)
    fileobj.seek(124)
    tst_str = fileobj.read(4)
    fileobj.seek(0)
    maj_ind = int(tst_str[2] == b'I'[0])
    maj_val = int(tst_str[maj_ind])
    min_val = int(tst_str[1 - maj_ind])
    ret = (maj_val, min_val)
    if maj_val in (1, 2):
        return ret
    raise ValueError('Unknown mat file type, version {}, {}'.format(*ret))