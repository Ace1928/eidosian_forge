from statsmodels.tools.sm_exceptions import CacheWriteWarning
from statsmodels.compat.pandas import cache_readonly as PandasCacheReadonly
import warnings
class CachedWritableAttribute(CachedAttribute):

    def __set__(self, obj, value):
        _cache = getattr(obj, self.cachename)
        name = self.name
        _cache[name] = value