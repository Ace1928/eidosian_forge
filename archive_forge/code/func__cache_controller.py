from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.middleware.cache import CacheMiddleware
from django.utils.cache import add_never_cache_headers, patch_cache_control
from django.utils.decorators import decorator_from_middleware_with_args
def _cache_controller(viewfunc):
    if iscoroutinefunction(viewfunc):

        async def _view_wrapper(request, *args, **kw):
            _check_request(request, 'cache_control')
            response = await viewfunc(request, *args, **kw)
            patch_cache_control(response, **kwargs)
            return response
    else:

        def _view_wrapper(request, *args, **kw):
            _check_request(request, 'cache_control')
            response = viewfunc(request, *args, **kw)
            patch_cache_control(response, **kwargs)
            return response
    return wraps(viewfunc)(_view_wrapper)