from heatclient import exc
import keystoneclient
from heat_integrationtests.functional import functional_base
def _is_sahara_deployed(self):
    try:
        self.identity_client.get_endpoint_url('data-processing', self.conf.region)
    except keystoneclient.exceptions.EndpointNotFound:
        return False
    return True