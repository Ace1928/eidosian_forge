import sys
import weakref
from sentry_sdk.api import continue_trace
from sentry_sdk._compat import reraise
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.tracing import (
from sentry_sdk.tracing_utils import should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class AioHttpIntegration(Integration):
    identifier = 'aiohttp'

    def __init__(self, transaction_style='handler_name'):
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError('Invalid value for transaction_style: %s (must be in %s)' % (transaction_style, TRANSACTION_STYLE_VALUES))
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        version = parse_version(AIOHTTP_VERSION)
        if version is None:
            raise DidNotEnable('Unparsable AIOHTTP version: {}'.format(AIOHTTP_VERSION))
        if version < (3, 4):
            raise DidNotEnable('AIOHTTP 3.4 or newer required.')
        if not HAS_REAL_CONTEXTVARS:
            raise DidNotEnable('The aiohttp integration for Sentry requires Python 3.7+  or aiocontextvars package.' + CONTEXTVARS_ERROR_MESSAGE)
        ignore_logger('aiohttp.server')
        old_handle = Application._handle

        async def sentry_app_handle(self, request, *args, **kwargs):
            hub = Hub.current
            if hub.get_integration(AioHttpIntegration) is None:
                return await old_handle(self, request, *args, **kwargs)
            weak_request = weakref.ref(request)
            with Hub(hub) as hub:
                with auto_session_tracking(hub, session_mode='request'):
                    with hub.configure_scope() as scope:
                        scope.clear_breadcrumbs()
                        scope.add_event_processor(_make_request_processor(weak_request))
                    headers = dict(request.headers)
                    transaction = continue_trace(headers, op=OP.HTTP_SERVER, name='generic AIOHTTP request', source=TRANSACTION_SOURCE_ROUTE)
                    with hub.start_transaction(transaction, custom_sampling_context={'aiohttp_request': request}):
                        try:
                            response = await old_handle(self, request)
                        except HTTPException as e:
                            transaction.set_http_status(e.status_code)
                            raise
                        except (asyncio.CancelledError, ConnectionResetError):
                            transaction.set_status('cancelled')
                            raise
                        except Exception:
                            reraise(*_capture_exception(hub))
                        transaction.set_http_status(response.status)
                        return response
        Application._handle = sentry_app_handle
        old_urldispatcher_resolve = UrlDispatcher.resolve

        async def sentry_urldispatcher_resolve(self, request):
            rv = await old_urldispatcher_resolve(self, request)
            hub = Hub.current
            integration = hub.get_integration(AioHttpIntegration)
            name = None
            try:
                if integration.transaction_style == 'handler_name':
                    name = transaction_from_function(rv.handler)
                elif integration.transaction_style == 'method_and_path_pattern':
                    route_info = rv.get_info()
                    pattern = route_info.get('path') or route_info.get('formatter')
                    name = '{} {}'.format(request.method, pattern)
            except Exception:
                pass
            if name is not None:
                with Hub.current.configure_scope() as scope:
                    scope.set_transaction_name(name, source=SOURCE_FOR_STYLE[integration.transaction_style])
            return rv
        UrlDispatcher.resolve = sentry_urldispatcher_resolve
        old_client_session_init = ClientSession.__init__

        def init(*args, **kwargs):
            hub = Hub.current
            if hub.get_integration(AioHttpIntegration) is None:
                return old_client_session_init(*args, **kwargs)
            client_trace_configs = list(kwargs.get('trace_configs') or ())
            trace_config = create_trace_config()
            client_trace_configs.append(trace_config)
            kwargs['trace_configs'] = client_trace_configs
            return old_client_session_init(*args, **kwargs)
        ClientSession.__init__ = init