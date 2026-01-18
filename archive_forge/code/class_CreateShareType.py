import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
from manilaclient.osc import utils
class CreateShareType(command.ShowOne):
    """Create new share type."""
    _description = _('Create new share type')

    def get_parser(self, prog_name):
        parser = super(CreateShareType, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', default=None, help=_('Share type name'))
        parser.add_argument('spec_driver_handles_share_servers', metavar='<spec_driver_handles_share_servers>', default=None, help=_("Required extra specification. Valid values are 'true' and 'false'"))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Share type description. Available only for microversion >= 2.41.'))
        parser.add_argument('--snapshot-support', metavar='<snapshot_support>', default=None, help=_('Boolean extra spec used for filtering of back ends by their capability to create share snapshots.'))
        parser.add_argument('--create-share-from-snapshot-support', metavar='<create_share_from_snapshot_support>', default=None, help=_('Boolean extra spec used for filtering of back ends by their capability to create shares from snapshots.'))
        parser.add_argument('--revert-to-snapshot-support', metavar='<revert_to_snapshot_support>', default=False, help=_('Boolean extra spec used for filtering of back ends by their capability to revert shares to snapshots. (Default is False).'))
        parser.add_argument('--mount-snapshot-support', metavar='<mount_snapshot_support>', default=False, help=_('Boolean extra spec used for filtering of back ends by their capability to mount share snapshots. (Default is False).'))
        parser.add_argument('--extra-specs', type=str, nargs='*', metavar='<key=value>', default=None, help=_("Extra specs key and value of share type that will be used for share type creation. OPTIONAL: Default=None. example --extra-specs  thin_provisioning='<is> True', replication_type=readable."))
        parser.add_argument('--public', metavar='<public>', default=True, help=_('Make type accessible to the public (default true).'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        kwargs = {'name': parsed_args.name}
        try:
            kwargs['spec_driver_handles_share_servers'] = strutils.bool_from_string(parsed_args.spec_driver_handles_share_servers, strict=True)
        except ValueError as e:
            msg = 'Argument spec_driver_handles_share_servers argument is not valid: %s' % str(e)
            raise exceptions.CommandError(msg)
        if parsed_args.description:
            if share_client.api_version.matches(api_versions.APIVersion('2.41'), api_versions.APIVersion()):
                kwargs['description'] = parsed_args.description
            else:
                raise exceptions.CommandError('Adding description to share type is only available with API microversion >= 2.41')
        if parsed_args.public:
            kwargs['is_public'] = strutils.bool_from_string(parsed_args.public, default=True)
        extra_specs = {}
        if parsed_args.extra_specs:
            for item in parsed_args.extra_specs:
                key, value = item.split('=', 1)
                if key == 'driver_handles_share_servers':
                    msg = "'driver_handles_share_servers' is already set via positional argument."
                    raise exceptions.CommandError(msg)
                else:
                    extra_specs = utils.extract_extra_specs(extra_specs, [item])
        for key in constants.BOOL_SPECS:
            value = getattr(parsed_args, key)
            if value:
                extra_specs = utils.extract_extra_specs(extra_specs, [key + '=' + value])
        kwargs['extra_specs'] = extra_specs
        share_type = share_client.share_types.create(**kwargs)
        formatted_type = format_share_type(share_type, parsed_args.formatter)
        return (ATTRIBUTES, oscutils.get_dict_properties(formatted_type._info, ATTRIBUTES))