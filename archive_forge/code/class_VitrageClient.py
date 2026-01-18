from vitrageclient import exceptions as exc
from keystoneauth1 import adapter as keystoneauth
from oslo_utils import importutils
class VitrageClient(keystoneauth.Adapter):

    def request(self, url, method, **kwargs):
        headers = kwargs.setdefault('headers', {})
        headers.setdefault('Accept', 'application/json')
        if profiler_web:
            headers.update(profiler_web.get_trace_id_headers())
        raise_exc = kwargs.pop('raise_exc', True)
        resp = super(VitrageClient, self).request(url, method, raise_exc=False, **kwargs)
        if raise_exc and resp.status_code >= 400:
            raise exc.from_response(resp, url, method)
        return resp