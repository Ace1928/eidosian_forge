import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class ListShareSecurityService(command.Lister):
    """List security services."""
    _description = _('List security services.')

    def get_parser(self, prog_name):
        parser = super(ListShareSecurityService, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', help=_('Display information from all projects (Admin only).'))
        parser.add_argument('--share-network', metavar='<share-network>', default=None, help=_('Filter results by share network name or ID.'))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Filter results by status.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Filter results by security service name.'))
        parser.add_argument('--type', metavar='<type>', default=None, help=_('Filter results by security service type.'))
        parser.add_argument('--user', metavar='<user', default=None, help=_('Filter results by security service user or group used by project.'))
        parser.add_argument('--dns-ip', metavar='<dns-ip>', default=None, help=_("Filter results by DNS IP address used inside project's network."))
        parser.add_argument('--ou', metavar='<ou>', default=None, help=_('Filter results by security service OU (Organizational Unit). Available only for microversion >= 2.44.'))
        parser.add_argument('--default-ad-site', metavar='<default_ad_site>', dest='default_ad_site', default=None, help=_('Filter results by security service default_ad_site. Available only for microversion >= 2.76.'))
        parser.add_argument('--server', metavar='<server>', default=None, help=_('Filter results by security service IP address or hostname.'))
        parser.add_argument('--domain', metavar='<domain>', default=None, help=_('Filter results by security service domain.'))
        parser.add_argument('--detail', action='store_true', help=_('Show detailed information about filtered security services.'))
        parser.add_argument('--limit', metavar='<num-security-services>', type=int, default=None, action=parseractions.NonNegativeAction, help=_('Limit the number of security services returned'))
        parser.add_argument('--marker', metavar='<security-service>', help=_('The last security service ID of the previous page'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        columns = ['ID', 'Name', 'Status', 'Type']
        if parsed_args.all_projects:
            columns.append('Project ID')
        if parsed_args.detail:
            columns.append('Share Networks')
        search_opts = {'all_tenants': parsed_args.all_projects, 'status': parsed_args.status, 'name': parsed_args.name, 'type': parsed_args.type, 'user': parsed_args.user, 'dns_ip': parsed_args.dns_ip, 'server': parsed_args.server, 'domain': parsed_args.domain, 'offset': parsed_args.marker, 'limit': parsed_args.limit}
        if parsed_args.ou and share_client.api_version >= api_versions.APIVersion('2.44'):
            search_opts['ou'] = parsed_args.ou
        elif parsed_args.ou:
            raise exceptions.CommandError(_('Filtering results by security service Organizational Unit is available only for microversion >= 2.44'))
        if parsed_args.default_ad_site and share_client.api_version >= api_versions.APIVersion('2.76'):
            search_opts['default_ad_site'] = parsed_args.default_ad_site
        elif parsed_args.default_ad_site:
            raise exceptions.CommandError(_('Filtering results by security service Default AD site is available only for microversion >= 2.76'))
        if parsed_args.share_network:
            search_opts['share_network_id'] = oscutils.find_resource(share_client.share_networks, parsed_args.share_network).id
        data = share_client.security_services.list(search_opts=search_opts, detailed=parsed_args.detail)
        return (columns, (oscutils.get_item_properties(s, columns) for s in data))