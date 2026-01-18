import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class CreateShareGroup(command.ShowOne):
    """Create new share group."""
    _description = _('Create new share group')

    def get_parser(self, prog_name):
        parser = super(CreateShareGroup, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Share group name'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Share group description.'))
        parser.add_argument('--share-types', metavar='<share-types>', nargs='+', default=[], help=_('Name or ID of share type(s).'))
        parser.add_argument('--share-group-type', metavar='<share-group-type>', default=None, help=_('Share group type name or ID of the share group to be created.'))
        parser.add_argument('--share-network', metavar='<share-network>', default=False, help=_('Specify share network name or id'))
        parser.add_argument('--source-share-group-snapshot', metavar='<source-share-group-snapshot>', default=False, help=_('Share group snapshot name or ID to create the share group from.'))
        parser.add_argument('--availability-zone', metavar='<availability-zone>', default=None, help=_('Optional availability zone in which group should be created'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share group creation'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_types = []
        for share_type in parsed_args.share_types:
            share_types.append(osc_utils.find_resource(share_client.share_types, share_type))
        share_group_type = None
        if parsed_args.share_group_type:
            share_group_type = osc_utils.find_resource(share_client.share_group_types, parsed_args.share_group_type).id
        share_network = None
        if parsed_args.share_network:
            share_network = osc_utils.find_resource(share_client.share_networks, parsed_args.share_network).id
        source_share_group_snapshot = None
        if parsed_args.source_share_group_snapshot:
            source_share_group_snapshot = osc_utils.find_resource(share_client.share_group_snapshots, parsed_args.source_share_group_snapshot).id
        body = {'name': parsed_args.name, 'description': parsed_args.description, 'share_types': share_types, 'share_group_type': share_group_type, 'share_network': share_network, 'source_share_group_snapshot': source_share_group_snapshot, 'availability_zone': parsed_args.availability_zone}
        share_group = share_client.share_groups.create(**body)
        if parsed_args.wait:
            if not osc_utils.wait_for_status(status_f=share_client.share_groups.get, res_id=share_group.id, success_status=['available']):
                LOG.error(_('ERROR: Share group is in error state.'))
            share_group = osc_utils.find_resource(share_client.share_groups, share_group.id)
        printable_share_group = share_group._info
        printable_share_group.pop('links', None)
        if printable_share_group.get('share_types'):
            if parsed_args.formatter == 'table':
                printable_share_group['share_types'] = '\n'.join(printable_share_group['share_types'])
        return self.dict2columns(printable_share_group)