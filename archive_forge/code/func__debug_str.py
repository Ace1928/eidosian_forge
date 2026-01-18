import stat
import time
from paramiko.common import x80000000, o700, o70, xffffffff
def _debug_str(self):
    out = '[ '
    if self.st_size is not None:
        out += 'size={} '.format(self.st_size)
    if self.st_uid is not None and self.st_gid is not None:
        out += 'uid={} gid={} '.format(self.st_uid, self.st_gid)
    if self.st_mode is not None:
        out += 'mode=' + oct(self.st_mode) + ' '
    if self.st_atime is not None and self.st_mtime is not None:
        out += 'atime={} mtime={} '.format(self.st_atime, self.st_mtime)
    for k, v in self.attr.items():
        out += '"{}"={!r} '.format(str(k), v)
    out += ']'
    return out