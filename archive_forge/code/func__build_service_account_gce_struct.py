import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _build_service_account_gce_struct(self, service_account, default_email='default', default_scope='devstorage.read_only'):
    """
         Helper to create Service Account dict.  Use
         _build_service_accounts_gce_list to create a list ready for the
         GCE API.

         :param: service_account: dictionarie containing email
                                  and list of scopes, e.g.
                                  [{'email':'default',
                                  'scopes':['compute', ...]}, ...]
                                  Scopes can either be full URLs or short
                                  names. If not provided, use the
                                  'default' service account email and a
                                  scope of 'devstorage.read_only'. Also
                                  accepts the aliases defined in
                                  'gcloud compute'.
        :type    service_account: ``dict`` or None

        :return: dict usable in GCE API call.
        :rtype:  ``dict``
        """
    if not isinstance(service_account, dict):
        raise ValueError("service_account not in the correct format,'%s - %s'" % (str(type(service_account)), str(service_account)))
    sa = {}
    if 'email' not in service_account:
        sa['email'] = default_email
    else:
        sa['email'] = service_account['email']
    if 'scopes' not in service_account:
        sa['scopes'] = [self.AUTH_URL + default_scope]
    else:
        ps = []
        for scope in service_account['scopes']:
            if scope.startswith(self.AUTH_URL):
                ps.append(scope)
            elif scope in self.SA_SCOPES_MAP:
                ps.append(self.AUTH_URL + self.SA_SCOPES_MAP[scope])
            else:
                ps.append(self.AUTH_URL + scope)
        sa['scopes'] = ps
    return sa