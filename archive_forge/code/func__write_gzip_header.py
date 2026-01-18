import struct, sys, time, os
import zlib
import builtins
import io
import _compression
def _write_gzip_header(self, compresslevel):
    self.fileobj.write(b'\x1f\x8b')
    self.fileobj.write(b'\x08')
    try:
        fname = os.path.basename(self.name)
        if not isinstance(fname, bytes):
            fname = fname.encode('latin-1')
        if fname.endswith(b'.gz'):
            fname = fname[:-3]
    except UnicodeEncodeError:
        fname = b''
    flags = 0
    if fname:
        flags = FNAME
    self.fileobj.write(chr(flags).encode('latin-1'))
    mtime = self._write_mtime
    if mtime is None:
        mtime = time.time()
    write32u(self.fileobj, int(mtime))
    if compresslevel == _COMPRESS_LEVEL_BEST:
        xfl = b'\x02'
    elif compresslevel == _COMPRESS_LEVEL_FAST:
        xfl = b'\x04'
    else:
        xfl = b'\x00'
    self.fileobj.write(xfl)
    self.fileobj.write(b'\xff')
    if fname:
        self.fileobj.write(fname + b'\x00')