import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class AdoptShare(command.ShowOne):
    """Adopt share not handled by Manila (Admin only)."""
    _description = _('Adopt a share')

    def get_parser(self, prog_name):
        parser = super(AdoptShare, self).get_parser(prog_name)
        parser.add_argument('service_host', metavar='<service-host>', help=_('Service host: some.host@driver#pool.'))
        parser.add_argument('protocol', metavar='<protocol>', help=_('Protocol of the share to manage, such as NFS or CIFS.'))
        parser.add_argument('export_path', metavar='<export-path>', help=_('Share export path, NFS share such as: 10.0.0.1:/example_path, CIFS share such as: \\\\10.0.0.1\\example_cifs_share.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Optional share name. (Default=None)'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Optional share description. (Default=None)'))
        parser.add_argument('--share-type', metavar='<share-type>', default=None, help=_('Optional share type assigned to share. (Default=None)'))
        parser.add_argument('--driver-options', type=str, nargs='*', metavar='<key=value>', default=None, help=_('Optional driver options as key=value pairs (Default=None).'))
        parser.add_argument('--public', action='store_true', help=_('Level of visibility for share. Defines whether other projects are able to see it or not. Available only for microversion >= 2.8. (Default=False)'))
        parser.add_argument('--share-server-id', metavar='<share-server-id>', help=_('Share server associated with share when using a share type with "driver_handles_share_servers" extra_spec set to True. Available only for microversion >= 2.49. (Default=None)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait until share is adopted'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        kwargs = {'service_host': parsed_args.service_host, 'protocol': parsed_args.protocol, 'export_path': parsed_args.export_path, 'name': parsed_args.name, 'description': parsed_args.description}
        share_type = None
        if parsed_args.share_type:
            share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type).id
            kwargs['share_type'] = share_type
        driver_options = None
        if parsed_args.driver_options:
            driver_options = utils.extract_properties(parsed_args.driver_options)
            kwargs['driver_options'] = driver_options
        if parsed_args.public:
            if share_client.api_version >= api_versions.APIVersion('2.8'):
                kwargs['public'] = True
            else:
                raise exceptions.CommandError('Setting share visibility while adopting a share is available only for API microversion >= 2.8')
        if parsed_args.share_server_id:
            if share_client.api_version >= api_versions.APIVersion('2.49'):
                kwargs['share_server_id'] = parsed_args.share_server_id
            else:
                raise exceptions.CommandError('Selecting a share server ID is available only for API microversion >= 2.49')
        share = share_client.shares.manage(**kwargs)
        if parsed_args.wait:
            if not oscutils.wait_for_status(status_f=share_client.shares.get, res_id=share.id, success_status=['available'], error_status=['manage_error', 'error']):
                LOG.error(_('ERROR: Share is in error state.'))
            share = apiutils.find_resource(share_client.shares, share.id)
        share._info.pop('links', None)
        return self.dict2columns(share._info)