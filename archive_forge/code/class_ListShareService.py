from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class ListShareService(command.Lister):
    """List share services (Admin only)."""
    _description = _('List share services (Admin only).')

    def get_parser(self, prog_name):
        parser = super(ListShareService, self).get_parser(prog_name)
        parser.add_argument('--host', metavar='<host>', default=None, help=_('Filter services by name of the host.'))
        parser.add_argument('--binary', metavar='<binary>', default=None, help=_('Filter services by the name of the service.'))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Filter results by status.'))
        parser.add_argument('--state', metavar='<state>', default=None, choices=['up', 'down'], help=_('Filter results by state.'))
        parser.add_argument('--zone', metavar='<zone>', default=None, help=_('Filter services by their availability zone.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        search_opts = {'host': parsed_args.host, 'binary': parsed_args.binary, 'status': parsed_args.status, 'state': parsed_args.state, 'zone': parsed_args.zone}
        services = share_client.services.list(search_opts=search_opts)
        columns = ['ID', 'Binary', 'Host', 'Zone', 'Status', 'State', 'Updated At']
        if share_client.api_version >= api_versions.APIVersion('2.83'):
            columns.append('Disabled Reason')
        data = (osc_utils.get_dict_properties(service._info, columns) for service in services)
        return (columns, data)