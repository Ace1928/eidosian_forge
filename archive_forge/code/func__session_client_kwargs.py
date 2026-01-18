import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
def _session_client_kwargs(self, session):
    kwargs = self._load_service_type_kwargs()
    kwargs['endpoint_override'] = None
    kwargs['session'] = session
    kwargs['api_version'] = None
    return kwargs