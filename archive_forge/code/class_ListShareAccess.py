import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class ListShareAccess(command.Lister):
    """List share access rules."""
    _description = _('List share access rule')

    def get_parser(self, prog_name):
        parser = super(ListShareAccess, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the share.'))
        parser.add_argument('--properties', type=str, nargs='*', metavar='<key=value>', default=None, help=_('Filters results by properties (key=value). OPTIONAL: Default=None. Available only for API microversion >= 2.45'))
        parser.add_argument('--access-type', metavar='<access_type>', default=None, help=_('Filter access rules by the access type.'))
        parser.add_argument('--access-key', metavar='<access_key>', default=None, help=_('Filter access rules by the access key.'))
        parser.add_argument('--access-to', metavar='<access_to>', default=None, help=_('Filter access rules by the access to field.'))
        parser.add_argument('--access-level', metavar='<access_level>', default=None, help=_('Filter access rules by the access level.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        access_type = parsed_args.access_type
        access_key = parsed_args.access_key
        access_to = parsed_args.access_to
        access_level = parsed_args.access_level
        extended_filter_keys = {'access_type': access_type, 'access_key': access_key, 'access_to': access_to, 'access_level': access_level}
        if any(extended_filter_keys.values()) and share_client.api_version < api_versions.APIVersion('2.82'):
            raise exceptions.CommandError('Filtering access rules by access_type, access_key, access_to and access_level is available starting from API version 2.82.')
        search_opts = {}
        if share_client.api_version >= api_versions.APIVersion('2.82'):
            for filter_key, filter_value in extended_filter_keys.items():
                if filter_value:
                    search_opts[filter_key] = filter_value
        if share_client.api_version >= api_versions.APIVersion('2.45'):
            if parsed_args.properties:
                search_opts = {'metadata': utils.extract_properties(parsed_args.properties)}
            access_list = share_client.share_access_rules.access_list(share, search_opts)
        elif parsed_args.properties:
            raise exceptions.CommandError('Filtering access rules by properties is supported only with API microversion 2.45 and beyond.')
        else:
            access_list = share.access_list()
        list_of_keys = ['ID', 'Access Type', 'Access To', 'Access Level', 'State']
        if share_client.api_version >= api_versions.APIVersion('2.21'):
            list_of_keys.append('Access Key')
        if share_client.api_version >= api_versions.APIVersion('2.33'):
            list_of_keys.append('Created At')
            list_of_keys.append('Updated At')
        values = (oscutils.get_item_properties(a, list_of_keys) for a in access_list)
        return (list_of_keys, values)