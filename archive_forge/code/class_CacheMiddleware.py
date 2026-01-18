from django.conf import settings
from django.core.cache import DEFAULT_CACHE_ALIAS, caches
from django.utils.cache import (
from django.utils.deprecation import MiddlewareMixin
class CacheMiddleware(UpdateCacheMiddleware, FetchFromCacheMiddleware):
    """
    Cache middleware that provides basic behavior for many simple sites.

    Also used as the hook point for the cache decorator, which is generated
    using the decorator-from-middleware utility.
    """

    def __init__(self, get_response, cache_timeout=None, page_timeout=None, **kwargs):
        super().__init__(get_response)
        try:
            key_prefix = kwargs['key_prefix']
            if key_prefix is None:
                key_prefix = ''
            self.key_prefix = key_prefix
        except KeyError:
            pass
        try:
            cache_alias = kwargs['cache_alias']
            if cache_alias is None:
                cache_alias = DEFAULT_CACHE_ALIAS
            self.cache_alias = cache_alias
        except KeyError:
            pass
        if cache_timeout is not None:
            self.cache_timeout = cache_timeout
        self.page_timeout = page_timeout