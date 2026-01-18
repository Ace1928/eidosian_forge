from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _get_compression_parameter(params, param):
    result = ffi.new('int *')
    zresult = lib.ZSTD_CCtxParams_getParameter(params, param, result)
    if lib.ZSTD_isError(zresult):
        raise ZstdError('unable to get compression context parameter: %s' % _zstd_error(zresult))
    return result[0]