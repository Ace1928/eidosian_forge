import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class WorldWideDNSConnection(ConnectionUserAndKey):
    host = 'www.worldwidedns.net'
    responseCls = WorldWideDNSResponse

    def add_default_params(self, params):
        """
        Add parameters that are necessary for every request

        This method adds ``NAME`` and ``PASSWORD`` to
        the request.
        """
        params['NAME'] = self.user_id
        params['PASSWORD'] = self.key
        reseller_id = getattr(self, 'reseller_id', None)
        if reseller_id:
            params['ID'] = reseller_id
        return params