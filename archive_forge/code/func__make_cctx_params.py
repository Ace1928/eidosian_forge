from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
def _make_cctx_params(params):
    res = lib.ZSTD_createCCtxParams()
    if res == ffi.NULL:
        raise MemoryError()
    res = ffi.gc(res, lib.ZSTD_freeCCtxParams)
    attrs = [(lib.ZSTD_c_format, params.format), (lib.ZSTD_c_compressionLevel, params.compression_level), (lib.ZSTD_c_windowLog, params.window_log), (lib.ZSTD_c_hashLog, params.hash_log), (lib.ZSTD_c_chainLog, params.chain_log), (lib.ZSTD_c_searchLog, params.search_log), (lib.ZSTD_c_minMatch, params.min_match), (lib.ZSTD_c_targetLength, params.target_length), (lib.ZSTD_c_strategy, params.strategy), (lib.ZSTD_c_contentSizeFlag, params.write_content_size), (lib.ZSTD_c_checksumFlag, params.write_checksum), (lib.ZSTD_c_dictIDFlag, params.write_dict_id), (lib.ZSTD_c_nbWorkers, params.threads), (lib.ZSTD_c_jobSize, params.job_size), (lib.ZSTD_c_overlapLog, params.overlap_log), (lib.ZSTD_c_forceMaxWindow, params.force_max_window), (lib.ZSTD_c_enableLongDistanceMatching, params.enable_ldm), (lib.ZSTD_c_ldmHashLog, params.ldm_hash_log), (lib.ZSTD_c_ldmMinMatch, params.ldm_min_match), (lib.ZSTD_c_ldmBucketSizeLog, params.ldm_bucket_size_log), (lib.ZSTD_c_ldmHashRateLog, params.ldm_hash_rate_log)]
    for param, value in attrs:
        _set_compression_parameter(res, param, value)
    return res