import collections
from requests import compat
def _get_proxy_information(response):
    if getattr(response.connection, 'proxy_manager', False):
        proxy_info = {}
        request_url = response.request.url
        if request_url.startswith('https://'):
            proxy_info['method'] = 'CONNECT'
        proxy_info['request_path'] = request_url
        return proxy_info
    return None