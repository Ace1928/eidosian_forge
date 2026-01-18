import os
from cliff import command
from cliff import lister
from cliff import show
from barbicanclient.v1 import secrets
class DeleteConsumer(command.Command):
    """Delete a consumer from a secret."""

    def get_parser(self, prog_name):
        parser = super(DeleteConsumer, self).get_parser(prog_name)
        parser.add_argument('URI', help='The URI reference for the secret')
        parser.add_argument('--service-type-name', '-s', required=True, help='the service that is consuming the secret')
        parser.add_argument('--resource-type', '-t', required=True, help='the type of resource that is consuming the secret')
        parser.add_argument('--resource-id', '-i', required=True, help='the id of the resource that is consuming the secret')
        return parser

    def take_action(self, args):
        self.app.client_manager.key_manager.secrets.remove_consumer(args.URI, args.service_type_name, args.resource_type, args.resource_id)