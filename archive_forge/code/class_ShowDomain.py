import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowDomain(command.ShowOne):
    _description = _('Display domain details')

    def get_parser(self, prog_name):
        parser = super(ShowDomain, self).get_parser(prog_name)
        parser.add_argument('domain', metavar='<domain>', help=_('Domain to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        domain_str = common._get_token_resource(identity_client, 'domain', parsed_args.domain)
        domain = utils.find_resource(identity_client.domains, domain_str)
        domain._info.pop('links')
        return zip(*sorted(domain._info.items()))