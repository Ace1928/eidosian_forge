from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdCompressionParameters(object):
    """Low-level zstd compression parameters.

    This type represents a collection of parameters to control how zstd
    compression is performed.

    Instances can be constructed from raw parameters or derived from a
    base set of defaults specified from a compression level (recommended)
    via :py:meth:`ZstdCompressionParameters.from_level`.

    >>> # Derive compression settings for compression level 7.
    >>> params = zstandard.ZstdCompressionParameters.from_level(7)

    >>> # With an input size of 1MB
    >>> params = zstandard.ZstdCompressionParameters.from_level(7, source_size=1048576)

    Using ``from_level()``, it is also possible to override individual compression
    parameters or to define additional settings that aren't automatically derived.
    e.g.:

    >>> params = zstandard.ZstdCompressionParameters.from_level(4, window_log=10)
    >>> params = zstandard.ZstdCompressionParameters.from_level(5, threads=4)

    Or you can define low-level compression settings directly:

    >>> params = zstandard.ZstdCompressionParameters(window_log=12, enable_ldm=True)

    Once a ``ZstdCompressionParameters`` instance is obtained, it can be used to
    configure a compressor:

    >>> cctx = zstandard.ZstdCompressor(compression_params=params)

    Some of these are very low-level settings. It may help to consult the official
    zstandard documentation for their behavior. Look for the ``ZSTD_p_*`` constants
    in ``zstd.h`` (https://github.com/facebook/zstd/blob/dev/lib/zstd.h).
    """

    @staticmethod
    def from_level(level, source_size=0, dict_size=0, **kwargs):
        """Create compression parameters from a compression level.

        :param level:
           Integer compression level.
        :param source_size:
           Integer size in bytes of source to be compressed.
        :param dict_size:
           Integer size in bytes of compression dictionary to use.
        :return:
           :py:class:`ZstdCompressionParameters`
        """
        params = lib.ZSTD_getCParams(level, source_size, dict_size)
        args = {'window_log': 'windowLog', 'chain_log': 'chainLog', 'hash_log': 'hashLog', 'search_log': 'searchLog', 'min_match': 'minMatch', 'target_length': 'targetLength', 'strategy': 'strategy'}
        for arg, attr in args.items():
            if arg not in kwargs:
                kwargs[arg] = getattr(params, attr)
        return ZstdCompressionParameters(**kwargs)

    def __init__(self, format=0, compression_level=0, window_log=0, hash_log=0, chain_log=0, search_log=0, min_match=0, target_length=0, strategy=-1, write_content_size=1, write_checksum=0, write_dict_id=0, job_size=0, overlap_log=-1, force_max_window=0, enable_ldm=0, ldm_hash_log=0, ldm_min_match=0, ldm_bucket_size_log=0, ldm_hash_rate_log=-1, threads=0):
        params = lib.ZSTD_createCCtxParams()
        if params == ffi.NULL:
            raise MemoryError()
        params = ffi.gc(params, lib.ZSTD_freeCCtxParams)
        self._params = params
        if threads < 0:
            threads = _cpu_count()
        _set_compression_parameter(params, lib.ZSTD_c_nbWorkers, threads)
        _set_compression_parameter(params, lib.ZSTD_c_format, format)
        _set_compression_parameter(params, lib.ZSTD_c_compressionLevel, compression_level)
        _set_compression_parameter(params, lib.ZSTD_c_windowLog, window_log)
        _set_compression_parameter(params, lib.ZSTD_c_hashLog, hash_log)
        _set_compression_parameter(params, lib.ZSTD_c_chainLog, chain_log)
        _set_compression_parameter(params, lib.ZSTD_c_searchLog, search_log)
        _set_compression_parameter(params, lib.ZSTD_c_minMatch, min_match)
        _set_compression_parameter(params, lib.ZSTD_c_targetLength, target_length)
        if strategy == -1:
            strategy = 0
        _set_compression_parameter(params, lib.ZSTD_c_strategy, strategy)
        _set_compression_parameter(params, lib.ZSTD_c_contentSizeFlag, write_content_size)
        _set_compression_parameter(params, lib.ZSTD_c_checksumFlag, write_checksum)
        _set_compression_parameter(params, lib.ZSTD_c_dictIDFlag, write_dict_id)
        _set_compression_parameter(params, lib.ZSTD_c_jobSize, job_size)
        if overlap_log == -1:
            overlap_log = 0
        _set_compression_parameter(params, lib.ZSTD_c_overlapLog, overlap_log)
        _set_compression_parameter(params, lib.ZSTD_c_forceMaxWindow, force_max_window)
        _set_compression_parameter(params, lib.ZSTD_c_enableLongDistanceMatching, enable_ldm)
        _set_compression_parameter(params, lib.ZSTD_c_ldmHashLog, ldm_hash_log)
        _set_compression_parameter(params, lib.ZSTD_c_ldmMinMatch, ldm_min_match)
        _set_compression_parameter(params, lib.ZSTD_c_ldmBucketSizeLog, ldm_bucket_size_log)
        if ldm_hash_rate_log == -1:
            ldm_hash_rate_log = 0
        _set_compression_parameter(params, lib.ZSTD_c_ldmHashRateLog, ldm_hash_rate_log)

    @property
    def format(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_format)

    @property
    def compression_level(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_compressionLevel)

    @property
    def window_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_windowLog)

    @property
    def hash_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_hashLog)

    @property
    def chain_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_chainLog)

    @property
    def search_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_searchLog)

    @property
    def min_match(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_minMatch)

    @property
    def target_length(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_targetLength)

    @property
    def strategy(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_strategy)

    @property
    def write_content_size(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_contentSizeFlag)

    @property
    def write_checksum(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_checksumFlag)

    @property
    def write_dict_id(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_dictIDFlag)

    @property
    def job_size(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_jobSize)

    @property
    def overlap_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_overlapLog)

    @property
    def force_max_window(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_forceMaxWindow)

    @property
    def enable_ldm(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_enableLongDistanceMatching)

    @property
    def ldm_hash_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_ldmHashLog)

    @property
    def ldm_min_match(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_ldmMinMatch)

    @property
    def ldm_bucket_size_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_ldmBucketSizeLog)

    @property
    def ldm_hash_rate_log(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_ldmHashRateLog)

    @property
    def threads(self):
        return _get_compression_parameter(self._params, lib.ZSTD_c_nbWorkers)

    def estimated_compression_context_size(self):
        """Estimated size in bytes needed to compress with these parameters."""
        return lib.ZSTD_estimateCCtxSize_usingCCtxParams(self._params)