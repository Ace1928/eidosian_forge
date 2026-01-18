import uuid
from keystoneauth1 import adapter
from designateclient import exceptions
class AdapterWithTimeout(adapter.Adapter):
    """adapter.Adapter wraps around a Session.

    The user can pass a timeout keyword that will apply only to
    the Designate Client, in order:

    - timeout keyword passed to ``request()``
    - timeout keyword passed to ``AdapterWithTimeout()``
    - timeout attribute on keystone session
    """

    def __init__(self, *args, **kw):
        self.timeout = kw.pop('timeout', None)
        super(self.__class__, self).__init__(*args, **kw)

    def request(self, *args, **kwargs):
        if self.timeout is not None:
            kwargs.setdefault('timeout', self.timeout)
        return super().request(*args, **kwargs)