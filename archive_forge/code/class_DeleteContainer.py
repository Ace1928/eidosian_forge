from cliff import command
from cliff import lister
from cliff import show
from barbicanclient.v1.containers import CertificateContainer
from barbicanclient.v1.containers import Container
from barbicanclient.v1.containers import RSAContainer
class DeleteContainer(command.Command):
    """Delete a container by providing its href."""

    def get_parser(self, prog_name):
        parser = super(DeleteContainer, self).get_parser(prog_name)
        parser.add_argument('URI', help='The URI reference for the container')
        return parser

    def take_action(self, args):
        self.app.client_manager.key_manager.containers.delete(args.URI)