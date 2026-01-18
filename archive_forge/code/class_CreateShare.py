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
class CreateShare(command.ShowOne):
    """Create a new share."""
    _description = _('Create new share')

    def get_parser(self, prog_name):
        parser = super(CreateShare, self).get_parser(prog_name)
        parser.add_argument('share_proto', metavar='<share_protocol>', help=_('Share protocol (NFS, CIFS, CephFS, GlusterFS or HDFS)'))
        parser.add_argument('size', metavar='<size>', type=int, help=_('Share size in GiB.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Optional share name. (Default=None)'))
        parser.add_argument('--snapshot-id', metavar='<snapshot-id>', default=None, help=_('Optional snapshot ID to create the share from. (Default=None)'))
        parser.add_argument('--property', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set a property to this share (repeat option to set multiple properties)'))
        parser.add_argument('--share-network', metavar='<network-info>', default=None, help=_('Optional network info ID or name.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Optional share description. (Default=None)'))
        parser.add_argument('--public', metavar='<public>', default=False, help=_('Level of visibility for share. Defines whether other tenants are able to see it or not. (Default = False)'))
        parser.add_argument('--share-type', metavar='<share-type>', default=None, help=_('The share type to create the share with. If not specified, unless creating from a snapshot, the default share type will be used.'))
        parser.add_argument('--availability-zone', metavar='<availability-zone>', default=None, help=_('Availability zone in which share should be created.'))
        parser.add_argument('--share-group', metavar='<share-group>', default=None, help=_('Optional share group name or ID in which to create the share. (Default=None).'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share creation'))
        parser.add_argument('--scheduler-hint', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set Scheduler hints for the share as key=value pairs, possible keys are same_host, different_host.(repeat option to set multiple hints)'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        if parsed_args.name:
            if parsed_args.name.capitalize() == 'None':
                raise apiclient_exceptions.CommandError("Share name cannot be with the value 'None'")
        share_type = None
        if parsed_args.share_type:
            share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type).id
        elif not parsed_args.snapshot_id:
            try:
                share_type = share_client.share_types.get(share_type='default').id
            except apiclient_exceptions.CommandError:
                msg = 'There is no default share type available. You must pick a valid share type to create a share.'
                raise exceptions.CommandError(msg)
        share_network = None
        if parsed_args.share_network:
            share_network = apiutils.find_resource(share_client.share_networks, parsed_args.share_network).id
        share_group = None
        if parsed_args.share_group:
            share_group = apiutils.find_resource(share_client.share_groups, parsed_args.share_group).id
        size = parsed_args.size
        snapshot_id = None
        if parsed_args.snapshot_id:
            snapshot = apiutils.find_resource(share_client.share_snapshots, parsed_args.snapshot_id)
            snapshot_id = snapshot.id
            size = max(size or 0, snapshot.size)
        scheduler_hints = {}
        if parsed_args.scheduler_hint:
            if share_client.api_version < api_versions.APIVersion('2.65'):
                raise exceptions.CommandError('Setting share scheduler hints for a share is available only for API microversion >= 2.65')
            else:
                scheduler_hints = utils.extract_key_value_options(parsed_args.scheduler_hint)
                same_host_hint_shares = scheduler_hints.get('same_host')
                different_host_hint_shares = scheduler_hints.get('different_host')
                if same_host_hint_shares:
                    same_host_hint_shares = [apiutils.find_resource(share_client.shares, sh).id for sh in same_host_hint_shares.split(',')]
                    scheduler_hints['same_host'] = ','.join(same_host_hint_shares)
                if different_host_hint_shares:
                    different_host_hint_shares = [apiutils.find_resource(share_client.shares, sh).id for sh in different_host_hint_shares.split(',')]
                    scheduler_hints['different_host'] = ','.join(different_host_hint_shares)
        body = {'share_proto': parsed_args.share_proto, 'size': size, 'snapshot_id': snapshot_id, 'name': parsed_args.name, 'description': parsed_args.description, 'metadata': parsed_args.property, 'share_network': share_network, 'share_type': share_type, 'is_public': parsed_args.public, 'availability_zone': parsed_args.availability_zone, 'share_group_id': share_group, 'scheduler_hints': scheduler_hints}
        share = share_client.shares.create(**body)
        if parsed_args.wait:
            if not oscutils.wait_for_status(status_f=share_client.shares.get, res_id=share.id, success_status=['available']):
                LOG.error(_('ERROR: Share is in error state.'))
            share = apiutils.find_resource(share_client.shares, share.id)
        printable_share = share._info
        printable_share.pop('links', None)
        printable_share.pop('shares_type', None)
        return self.dict2columns(printable_share)