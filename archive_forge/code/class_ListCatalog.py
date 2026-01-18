import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListCatalog(command.Lister):
    _description = _('List services in the service catalog')

    def take_action(self, parsed_args):
        auth_ref = self.app.client_manager.auth_ref
        if not auth_ref:
            raise exceptions.AuthorizationFailure('Only an authorized user may issue a new token.')
        data = auth_ref.service_catalog.catalog
        columns = ('Name', 'Type', 'Endpoints')
        return (columns, (utils.get_dict_properties(s, columns, formatters={'Endpoints': EndpointsColumn}) for s in data))