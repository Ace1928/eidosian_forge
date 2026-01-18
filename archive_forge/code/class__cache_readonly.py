from statsmodels.tools.sm_exceptions import CacheWriteWarning
from statsmodels.compat.pandas import cache_readonly as PandasCacheReadonly
import warnings
class _cache_readonly(property):
    """
    Decorator for CachedAttribute
    """

    def __init__(self, cachename=None):
        self.func = None
        self.cachename = cachename

    def __call__(self, func):
        return CachedAttribute(func, cachename=self.cachename)