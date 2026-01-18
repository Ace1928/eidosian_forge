from django.conf import settings
from django.core.cache import DEFAULT_CACHE_ALIAS, caches
from django.utils.cache import (
from django.utils.deprecation import MiddlewareMixin
class UpdateCacheMiddleware(MiddlewareMixin):
    """
    Response-phase cache middleware that updates the cache if the response is
    cacheable.

    Must be used as part of the two-part update/fetch cache middleware.
    UpdateCacheMiddleware must be the first piece of middleware in MIDDLEWARE
    so that it'll get called last during the response phase.
    """

    def __init__(self, get_response):
        super().__init__(get_response)
        self.cache_timeout = settings.CACHE_MIDDLEWARE_SECONDS
        self.page_timeout = None
        self.key_prefix = settings.CACHE_MIDDLEWARE_KEY_PREFIX
        self.cache_alias = settings.CACHE_MIDDLEWARE_ALIAS

    @property
    def cache(self):
        return caches[self.cache_alias]

    def _should_update_cache(self, request, response):
        return hasattr(request, '_cache_update_cache') and request._cache_update_cache

    def process_response(self, request, response):
        """Set the cache, if needed."""
        if not self._should_update_cache(request, response):
            return response
        if response.streaming or response.status_code not in (200, 304):
            return response
        if not request.COOKIES and response.cookies and has_vary_header(response, 'Cookie'):
            return response
        if 'private' in response.get('Cache-Control', ()):
            return response
        timeout = self.page_timeout
        if timeout is None:
            timeout = get_max_age(response)
            if timeout is None:
                timeout = self.cache_timeout
            elif timeout == 0:
                return response
        patch_response_headers(response, timeout)
        if timeout and response.status_code == 200:
            cache_key = learn_cache_key(request, response, timeout, self.key_prefix, cache=self.cache)
            if hasattr(response, 'render') and callable(response.render):
                response.add_post_render_callback(lambda r: self.cache.set(cache_key, r, timeout))
            else:
                self.cache.set(cache_key, response, timeout)
        return response