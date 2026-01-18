import debtcollector.renames
from neutronclient import client
from neutronclient.neutron import client as neutron_client
class ClientManager(object):
    """Manages access to API clients, including authentication."""
    neutron = ClientCache(neutron_client.make_client)
    quantum = neutron

    @debtcollector.renames.renamed_kwarg('tenant_id', 'project_id', replace=True)
    @debtcollector.renames.renamed_kwarg('tenant_name', 'project_name', replace=True)
    def __init__(self, token=None, url=None, auth_url=None, endpoint_type=None, project_name=None, project_id=None, username=None, user_id=None, password=None, region_name=None, api_version=None, auth_strategy=None, insecure=False, ca_cert=None, log_credentials=False, service_type=None, service_name=None, timeout=None, retries=0, raise_errors=True, session=None, auth=None):
        self._token = token
        self._url = url
        self._auth_url = auth_url
        self._service_type = service_type
        self._service_name = service_name
        self._endpoint_type = endpoint_type
        self._project_name = project_name
        self._project_id = project_id
        self._username = username
        self._user_id = user_id
        self._password = password
        self._region_name = region_name
        self._api_version = api_version
        self._service_catalog = None
        self._auth_strategy = auth_strategy
        self._insecure = insecure
        self._ca_cert = ca_cert
        self._log_credentials = log_credentials
        self._timeout = timeout
        self._retries = retries
        self._raise_errors = raise_errors
        self._session = session
        self._auth = auth
        return

    def initialize(self):
        if not self._url:
            httpclient = client.construct_http_client(username=self._username, user_id=self._user_id, project_name=self._project_name, project_id=self._project_id, password=self._password, region_name=self._region_name, auth_url=self._auth_url, service_type=self._service_type, service_name=self._service_name, endpoint_type=self._endpoint_type, insecure=self._insecure, ca_cert=self._ca_cert, timeout=self._timeout, session=self._session, auth=self._auth, log_credentials=self._log_credentials)
            httpclient.authenticate()
            self._token = httpclient.auth_token
            self._url = httpclient.endpoint_url