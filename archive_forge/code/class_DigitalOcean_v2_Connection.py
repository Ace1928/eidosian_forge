from libcloud.utils.py3 import httplib, parse_qs, urlparse
from libcloud.common.base import BaseDriver, JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class DigitalOcean_v2_Connection(ConnectionKey):
    """
    Connection class for the DigitalOcean (v2) driver.
    """
    host = 'api.digitalocean.com'
    responseCls = DigitalOcean_v2_Response

    def add_default_headers(self, headers):
        """
        Add headers that are necessary for every request

        This method adds ``token`` to the request.
        """
        headers['Authorization'] = 'Bearer %s' % self.key
        headers['Content-Type'] = 'application/json'
        return headers

    def add_default_params(self, params):
        """
        Add parameters that are necessary for every request

        This method adds ``per_page`` to the request to reduce the total
        number of paginated requests to the API.
        """
        params['per_page'] = self.driver.ex_per_page
        return params