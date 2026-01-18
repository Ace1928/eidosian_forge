import logging
from blazarclient import command
class ShowFloatingIP(command.ShowCommand):
    """Show floating IP details."""
    resource = 'floatingip'
    allow_names = False
    json_indent = 4
    log = logging.getLogger(__name__ + '.ShowFloatingIP')

    def get_parser(self, prog_name):
        parser = super(ShowFloatingIP, self).get_parser(prog_name)
        if self.allow_names:
            help_str = 'ID or name of %s to look up'
        else:
            help_str = 'ID of %s to look up'
        parser.add_argument('id', metavar=self.resource.upper(), help=help_str % self.resource)
        return parser