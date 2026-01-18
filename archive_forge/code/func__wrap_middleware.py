from django import VERSION as DJANGO_VERSION
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import (
def _wrap_middleware(middleware, middleware_name):
    from sentry_sdk.integrations.django import DjangoIntegration

    def _check_middleware_span(old_method):
        hub = Hub.current
        integration = hub.get_integration(DjangoIntegration)
        if integration is None or not integration.middleware_spans:
            return None
        function_name = transaction_from_function(old_method)
        description = middleware_name
        function_basename = getattr(old_method, '__name__', None)
        if function_basename:
            description = '{}.{}'.format(description, function_basename)
        middleware_span = hub.start_span(op=OP.MIDDLEWARE_DJANGO, description=description)
        middleware_span.set_tag('django.function_name', function_name)
        middleware_span.set_tag('django.middleware_name', middleware_name)
        return middleware_span

    def _get_wrapped_method(old_method):
        with capture_internal_exceptions():

            def sentry_wrapped_method(*args, **kwargs):
                middleware_span = _check_middleware_span(old_method)
                if middleware_span is None:
                    return old_method(*args, **kwargs)
                with middleware_span:
                    return old_method(*args, **kwargs)
            try:
                sentry_wrapped_method = wraps(old_method)(sentry_wrapped_method)
                sentry_wrapped_method.__self__ = old_method.__self__
            except Exception:
                pass
            return sentry_wrapped_method
        return old_method

    class SentryWrappingMiddleware(_asgi_middleware_mixin_factory(_check_middleware_span)):
        async_capable = getattr(middleware, 'async_capable', False)

        def __init__(self, get_response=None, *args, **kwargs):
            if get_response:
                self._inner = middleware(get_response, *args, **kwargs)
            else:
                self._inner = middleware(*args, **kwargs)
            self.get_response = get_response
            self._call_method = None
            if self.async_capable:
                super(SentryWrappingMiddleware, self).__init__(get_response)

        def __getattr__(self, method_name):
            if method_name not in ('process_request', 'process_view', 'process_template_response', 'process_response', 'process_exception'):
                raise AttributeError()
            old_method = getattr(self._inner, method_name)
            rv = _get_wrapped_method(old_method)
            self.__dict__[method_name] = rv
            return rv

        def __call__(self, *args, **kwargs):
            if hasattr(self, 'async_route_check') and self.async_route_check():
                return self.__acall__(*args, **kwargs)
            f = self._call_method
            if f is None:
                self._call_method = f = self._inner.__call__
            middleware_span = _check_middleware_span(old_method=f)
            if middleware_span is None:
                return f(*args, **kwargs)
            with middleware_span:
                return f(*args, **kwargs)
    for attr in ('__name__', '__module__', '__qualname__'):
        if hasattr(middleware, attr):
            setattr(SentryWrappingMiddleware, attr, getattr(middleware, attr))
    return SentryWrappingMiddleware