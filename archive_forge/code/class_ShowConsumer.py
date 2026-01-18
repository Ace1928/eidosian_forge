import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowConsumer(command.ShowOne):
    _description = _('Display consumer details')

    def get_parser(self, prog_name):
        parser = super(ShowConsumer, self).get_parser(prog_name)
        parser.add_argument('consumer', metavar='<consumer>', help=_('Consumer to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        consumer = utils.find_resource(identity_client.oauth1.consumers, parsed_args.consumer)
        consumer._info.pop('links', None)
        return zip(*sorted(consumer._info.items()))