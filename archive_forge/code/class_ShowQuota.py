import abc
import argparse
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ShowQuota(ShowQuotaBase):
    """Show quotas for a given tenant."""

    def retrieve_data(self, tenant_id, neutron_client):
        return neutron_client.show_quota(tenant_id)