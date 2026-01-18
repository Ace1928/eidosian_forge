from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
@property
def _ddict(self):
    ddict = lib.ZSTD_createDDict_advanced(self._data, len(self._data), lib.ZSTD_dlm_byRef, self._dict_type, lib.ZSTD_defaultCMem)
    if ddict == ffi.NULL:
        raise ZstdError('could not create decompression dict')
    ddict = ffi.gc(ddict, lib.ZSTD_freeDDict, size=lib.ZSTD_sizeof_DDict(ddict))
    self.__dict__['_ddict'] = ddict
    return ddict