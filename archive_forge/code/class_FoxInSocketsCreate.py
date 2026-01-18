from neutronclient._i18n import _
from neutronclient.common import extension
class FoxInSocketsCreate(extension.ClientExtensionCreate, FoxInSocket):
    """Create a fox socket."""
    shell_command = 'fox-sockets-create'
    list_columns = ['id', 'name']

    def add_known_arguments(self, parser):
        _add_updatable_args(parser)

    def args2body(self, parsed_args):
        body = {}
        client = self.get_client()
        _updatable_args2body(parsed_args, body, client)
        return {'fox_socket': body}