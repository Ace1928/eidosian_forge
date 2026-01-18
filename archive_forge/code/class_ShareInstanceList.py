import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
class ShareInstanceList(command.Lister):
    """List share instances."""
    _description = _('List share instances')

    def get_parser(self, prog_name):
        parser = super(ShareInstanceList, self).get_parser(prog_name)
        parser.add_argument('--share', metavar='<share>', default=None, help=_('Name or ID of the share to list instances for.'))
        parser.add_argument('--export-location', metavar='<export-location>', default=None, help=_('Export location to list instances for.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        instances = []
        kwargs = {}
        if parsed_args.share:
            share = osc_utils.find_resource(share_client.shares, parsed_args.share)
            instances = share_client.shares.list_instances(share)
        else:
            if share_client.api_version < api_versions.APIVersion('2.35'):
                if parsed_args.export_location:
                    raise exceptions.CommandError('Filtering by export location is only available with manila API version >= 2.35')
            elif parsed_args.export_location:
                kwargs['export_location'] = parsed_args.export_location
            instances = share_client.share_instances.list(**kwargs)
        columns = ['ID', 'Share ID', 'Host', 'Status', 'Availability Zone', 'Share Network ID', 'Share Server ID', 'Share Type ID']
        data = (osc_utils.get_dict_properties(instance._info, columns) for instance in instances)
        return (columns, data)