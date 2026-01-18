from openstack import resource
def _request_kwargs(self, prepend_key=True, base_path=None):
    request = self._prepare_request(requires_id=False, prepend_key=prepend_key, base_path=base_path)
    headers = {'Content-Type': 'text/plain'}
    kwargs = {'data': self.definition}
    scope = '?scope=%s' % self.scope
    uri = request.url + scope
    request.headers.update(headers)
    return dict(url=uri, json=None, headers=request.headers, **kwargs)