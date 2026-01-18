import os
import warnings
import requests
from keystoneauth1 import _fair_semaphore
from keystoneauth1 import session
class LegacyJsonAdapter(Adapter):
    """Make something that looks like an old HTTPClient.

    A common case when using an adapter is that we want an interface similar to
    the HTTPClients of old which returned the body as JSON as well.

    You probably don't want this if you are starting from scratch.
    """

    def request(self, *args, **kwargs):
        headers = kwargs.setdefault('headers', {})
        headers.setdefault('Accept', 'application/json')
        try:
            kwargs['json'] = kwargs.pop('body')
        except KeyError:
            pass
        resp = super(LegacyJsonAdapter, self).request(*args, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            body = None
        return (resp, body)