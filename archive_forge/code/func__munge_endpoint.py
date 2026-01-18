from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
import simplejson as json
from osc_lib import exceptions
from osc_lib.i18n import _
def _munge_endpoint(self, endpoint):
    """Hook to allow subclasses to massage the passed-in endpoint

        Hook to massage passed-in endpoints from arbitrary sources,
        including direct user input.  By default just remove trailing
        '/' as all of our path info strings start with '/' and not all
        services can handle '//' in their URLs.

        Some subclasses will override this to do additional work, most
        likely with regard to API versions.

        :param string endpoint: The service endpoint, generally direct
                                from the service catalog.
        :return: The modified endpoint
        """
    if isinstance(endpoint, str):
        return endpoint.rstrip('/')
    else:
        return endpoint