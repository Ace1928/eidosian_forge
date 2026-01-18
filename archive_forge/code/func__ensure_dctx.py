from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _ensure_dctx(self, load_dict=True):
    lib.ZSTD_DCtx_reset(self._dctx, lib.ZSTD_reset_session_only)
    if self._max_window_size:
        zresult = lib.ZSTD_DCtx_setMaxWindowSize(self._dctx, self._max_window_size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('unable to set max window size: %s' % _zstd_error(zresult))
    zresult = lib.ZSTD_DCtx_setParameter(self._dctx, lib.ZSTD_d_format, self._format)
    if lib.ZSTD_isError(zresult):
        raise ZstdError('unable to set decoding format: %s' % _zstd_error(zresult))
    if self._dict_data and load_dict:
        zresult = lib.ZSTD_DCtx_refDDict(self._dctx, self._dict_data._ddict)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('unable to reference prepared dictionary: %s' % _zstd_error(zresult))