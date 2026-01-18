from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
class NeutronConstraint(constraints.BaseCustomConstraint):
    expected_exceptions = (qe.NeutronClientException, exception.EntityNotFound)
    resource_name = None
    extension = None

    def validate_with_client(self, client, value):
        neutron_plugin = client.client_plugin(CLIENT_NAME)
        if self.extension and (not neutron_plugin.has_extension(self.extension)):
            raise exception.EntityNotFound(entity='neutron extension', name=self.extension)
        neutron_plugin.find_resourceid_by_name_or_id(self.resource_name, value)