import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetConsumer(command.Command):
    _description = _('Set consumer properties')

    def get_parser(self, prog_name):
        parser = super(SetConsumer, self).get_parser(prog_name)
        parser.add_argument('consumer', metavar='<consumer>', help=_('Consumer to modify'))
        parser.add_argument('--description', metavar='<description>', help=_('New consumer description'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        consumer = utils.find_resource(identity_client.oauth1.consumers, parsed_args.consumer)
        kwargs = {}
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        consumer = identity_client.oauth1.consumers.update(consumer.id, **kwargs)