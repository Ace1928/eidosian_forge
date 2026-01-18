import requests
import tempfile
from oslo_config import cfg
from oslo_config import sources
def _fetch_uri(self, uri, ca_path, client_cert, client_key, timeout):
    verify = ca_path if ca_path else True
    cert = (client_cert, client_key) if client_cert and client_key else client_cert
    with requests.get(uri, verify=verify, cert=cert, timeout=timeout) as response:
        response.raise_for_status()
        return response.text