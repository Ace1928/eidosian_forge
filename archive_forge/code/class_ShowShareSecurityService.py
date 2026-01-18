import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class ShowShareSecurityService(command.ShowOne):
    """Show security service."""
    _description = _('Show security service.')

    def get_parser(self, prog_name):
        parser = super(ShowShareSecurityService, self).get_parser(prog_name)
        parser.add_argument('security_service', metavar='<security-service>', help=_('Security service name or ID to show.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        security_service = oscutils.find_resource(share_client.security_services, parsed_args.security_service)
        data = security_service._info
        if parsed_args.formatter == 'table':
            if 'share_networks' in data.keys():
                data['share_networks'] = '\n'.join(data['share_networks'])
        return self.dict2columns(data)