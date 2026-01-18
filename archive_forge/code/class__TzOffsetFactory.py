from datetime import timedelta
import weakref
from collections import OrderedDict
from six.moves import _thread
class _TzOffsetFactory(_TzFactory):

    def __init__(cls, *args, **kwargs):
        cls.__instances = weakref.WeakValueDictionary()
        cls.__strong_cache = OrderedDict()
        cls.__strong_cache_size = 8
        cls._cache_lock = _thread.allocate_lock()

    def __call__(cls, name, offset):
        if isinstance(offset, timedelta):
            key = (name, offset.total_seconds())
        else:
            key = (name, offset)
        instance = cls.__instances.get(key, None)
        if instance is None:
            instance = cls.__instances.setdefault(key, cls.instance(name, offset))
        with cls._cache_lock:
            cls.__strong_cache[key] = cls.__strong_cache.pop(key, instance)
            if len(cls.__strong_cache) > cls.__strong_cache_size:
                cls.__strong_cache.popitem(last=False)
        return instance