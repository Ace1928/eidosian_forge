def _set_default_headers(self, kwargs):
    headers = kwargs.get('headers', {})
    for k, v in self.DEFAULT_HEADERS.items():
        if k not in headers:
            headers[k] = v
    kwargs['headers'] = headers
    return kwargs