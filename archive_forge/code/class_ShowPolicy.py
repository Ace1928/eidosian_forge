import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowPolicy(command.ShowOne):
    _description = _('Display policy details')

    def get_parser(self, prog_name):
        parser = super(ShowPolicy, self).get_parser(prog_name)
        parser.add_argument('policy', metavar='<policy>', help=_('Policy to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        policy = utils.find_resource(identity_client.policies, parsed_args.policy)
        policy._info.pop('links')
        policy._info.update({'rules': policy._info.pop('blob')})
        return zip(*sorted(policy._info.items()))