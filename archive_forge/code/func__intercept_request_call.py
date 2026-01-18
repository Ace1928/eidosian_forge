from . import ws_client
def _intercept_request_call(*args, **kwargs):
    try:
        config = func.__self__.api_client.configuration
    except AttributeError:
        config = func.__self__.api_client.config
    return ws_client.websocket_call(config, *args, **kwargs)