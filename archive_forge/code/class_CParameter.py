from collections import namedtuple
from enum import IntEnum
from functools import lru_cache
from ._zstd import *
from . import _zstd
class CParameter(IntEnum):
    """Compression parameters"""
    compressionLevel = _zstd._ZSTD_c_compressionLevel
    windowLog = _zstd._ZSTD_c_windowLog
    hashLog = _zstd._ZSTD_c_hashLog
    chainLog = _zstd._ZSTD_c_chainLog
    searchLog = _zstd._ZSTD_c_searchLog
    minMatch = _zstd._ZSTD_c_minMatch
    targetLength = _zstd._ZSTD_c_targetLength
    strategy = _zstd._ZSTD_c_strategy
    enableLongDistanceMatching = _zstd._ZSTD_c_enableLongDistanceMatching
    ldmHashLog = _zstd._ZSTD_c_ldmHashLog
    ldmMinMatch = _zstd._ZSTD_c_ldmMinMatch
    ldmBucketSizeLog = _zstd._ZSTD_c_ldmBucketSizeLog
    ldmHashRateLog = _zstd._ZSTD_c_ldmHashRateLog
    contentSizeFlag = _zstd._ZSTD_c_contentSizeFlag
    checksumFlag = _zstd._ZSTD_c_checksumFlag
    dictIDFlag = _zstd._ZSTD_c_dictIDFlag
    nbWorkers = _zstd._ZSTD_c_nbWorkers
    jobSize = _zstd._ZSTD_c_jobSize
    overlapLog = _zstd._ZSTD_c_overlapLog

    @lru_cache(maxsize=None)
    def bounds(self):
        """Return lower and upper bounds of a compression parameter, both inclusive."""
        return _zstd._get_param_bounds(1, self.value)