import logging
from blazarclient import command
from blazarclient import exception
class ListHosts(command.ListCommand):
    """Print a list of hosts."""
    resource = 'host'
    log = logging.getLogger(__name__ + '.ListHosts')
    list_columns = ['id', 'hypervisor_hostname', 'vcpus', 'memory_mb', 'local_gb']

    def get_parser(self, prog_name):
        parser = super(ListHosts, self).get_parser(prog_name)
        parser.add_argument('--sort-by', metavar='<host_column>', help='column name used to sort result', default='hypervisor_hostname')
        return parser