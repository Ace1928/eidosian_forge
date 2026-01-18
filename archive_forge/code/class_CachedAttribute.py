from statsmodels.tools.sm_exceptions import CacheWriteWarning
from statsmodels.compat.pandas import cache_readonly as PandasCacheReadonly
import warnings
class CachedAttribute:

    def __init__(self, func, cachename=None):
        self.fget = func
        self.name = func.__name__
        self.cachename = cachename or '_cache'

    def __get__(self, obj, type=None):
        if obj is None:
            return self.fget
        _cachename = self.cachename
        _cache = getattr(obj, _cachename, None)
        if _cache is None:
            setattr(obj, _cachename, {})
            _cache = getattr(obj, _cachename)
        name = self.name
        _cachedval = _cache.get(name, None)
        if _cachedval is None:
            _cachedval = self.fget(obj)
            _cache[name] = _cachedval
        return _cachedval

    def __set__(self, obj, value):
        errmsg = "The attribute '%s' cannot be overwritten" % self.name
        warnings.warn(errmsg, CacheWriteWarning)