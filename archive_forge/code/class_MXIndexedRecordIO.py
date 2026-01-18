from collections import namedtuple
from multiprocessing import current_process
import ctypes
import struct
import numbers
import numpy as np
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
class MXIndexedRecordIO(MXRecordIO):
    """Reads/writes `RecordIO` data format, supporting random access.

    Examples
    ---------
    >>> for i in range(5):
    ...     record.write_idx(i, 'record_%d'%i)
    >>> record.close()
    >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
    >>> record.read_idx(3)
    record_3

    Parameters
    ----------
    idx_path : str
        Path to the index file.
    uri : str
        Path to the record file. Only supports seekable file types.
    flag : str
        'w' for write or 'r' for read.
    key_type : type
        Data type for keys.
    """

    def __init__(self, idx_path, uri, flag, key_type=int):
        self.idx_path = idx_path
        self.idx = {}
        self.keys = []
        self.key_type = key_type
        self.fidx = None
        super(MXIndexedRecordIO, self).__init__(uri, flag)

    def open(self):
        super(MXIndexedRecordIO, self).open()
        self.idx = {}
        self.keys = []
        self.fidx = open(self.idx_path, self.flag)
        if not self.writable:
            for line in iter(self.fidx.readline, ''):
                line = line.strip().split('\t')
                key = self.key_type(line[0])
                self.idx[key] = int(line[1])
                self.keys.append(key)

    def close(self):
        """Closes the record file."""
        if not self.is_open:
            return
        super(MXIndexedRecordIO, self).close()
        self.fidx.close()

    def __getstate__(self):
        """Override pickling behavior."""
        d = super(MXIndexedRecordIO, self).__getstate__()
        d['fidx'] = None
        return d

    def seek(self, idx):
        """Sets the current read pointer position.

        This function is internally called by `read_idx(idx)` to find the current
        reader pointer position. It doesn't return anything."""
        assert not self.writable
        self._check_pid(allow_reset=True)
        pos = ctypes.c_size_t(self.idx[idx])
        check_call(_LIB.MXRecordIOReaderSeek(self.handle, pos))

    def tell(self):
        """Returns the current position of write head.

        Examples
        ---------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> print(record.tell())
        0
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        ...     print(record.tell())
        16
        32
        48
        64
        80
        """
        assert self.writable
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOWriterTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def read_idx(self, idx):
        """Returns the record at given index.

        Examples
        ---------
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'w')
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()
        >>> record = mx.recordio.MXIndexedRecordIO('tmp.idx', 'tmp.rec', 'r')
        >>> record.read_idx(3)
        record_3
        """
        self.seek(idx)
        return self.read()

    def write_idx(self, idx, buf):
        """Inserts input record at given index.

        Examples
        ---------
        >>> for i in range(5):
        ...     record.write_idx(i, 'record_%d'%i)
        >>> record.close()

        Parameters
        ----------
        idx : int
            Index of a file.
        buf :
            Record to write.
        """
        key = self.key_type(idx)
        pos = self.tell()
        self.write(buf)
        self.fidx.write('%s\t%d\n' % (str(key), pos))
        self.idx[key] = pos
        self.keys.append(key)