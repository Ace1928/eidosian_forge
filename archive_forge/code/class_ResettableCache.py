from statsmodels.tools.sm_exceptions import CacheWriteWarning
from statsmodels.compat.pandas import cache_readonly as PandasCacheReadonly
import warnings
class ResettableCache(dict):
    """DO NOT USE. BACKWARD COMPAT ONLY"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self