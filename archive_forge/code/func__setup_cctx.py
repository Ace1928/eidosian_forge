from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _setup_cctx(self):
    zresult = lib.ZSTD_CCtx_setParametersUsingCCtxParams(self._cctx, self._params)
    if lib.ZSTD_isError(zresult):
        raise ZstdError('could not set compression parameters: %s' % _zstd_error(zresult))
    dict_data = self._dict_data
    if dict_data:
        if dict_data._cdict:
            zresult = lib.ZSTD_CCtx_refCDict(self._cctx, dict_data._cdict)
        else:
            zresult = lib.ZSTD_CCtx_loadDictionary_advanced(self._cctx, dict_data.as_bytes(), len(dict_data), lib.ZSTD_dlm_byRef, dict_data._dict_type)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('could not load compression dictionary: %s' % _zstd_error(zresult))