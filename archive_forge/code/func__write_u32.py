from collections import namedtuple
import warnings
def _write_u32(file, x):
    data = []
    for i in range(4):
        d, m = divmod(x, 256)
        data.insert(0, int(m))
        x = d
    file.write(bytes(data))