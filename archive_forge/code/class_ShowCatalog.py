import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowCatalog(command.ShowOne):
    _description = _('Display service catalog details')

    def get_parser(self, prog_name):
        parser = super(ShowCatalog, self).get_parser(prog_name)
        parser.add_argument('service', metavar='<service>', help=_('Service to display (type or name)'))
        return parser

    def take_action(self, parsed_args):
        auth_ref = self.app.client_manager.auth_ref
        if not auth_ref:
            raise exceptions.AuthorizationFailure('Only an authorized user may issue a new token.')
        data = None
        for service in auth_ref.service_catalog.catalog:
            if service.get('name') == parsed_args.service or service.get('type') == parsed_args.service:
                data = service.copy()
                data['endpoints'] = EndpointsColumn(data['endpoints'])
                if 'endpoints_links' in data:
                    data.pop('endpoints_links')
                break
        if not data:
            LOG.error(_('service %s not found\n'), parsed_args.service)
            return ((), ())
        return zip(*sorted(data.items()))