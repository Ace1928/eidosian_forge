from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
class ShowHost(command.ShowOne):
    """Show a host"""
    log = logging.getLogger(__name__ + '.ShowHost')

    def get_parser(self, prog_name):
        parser = super(ShowHost, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help='ID or name of the host to show.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        host = parsed_args.host
        host = client.hosts.get(host)
        columns = _host_columns(host)
        return (columns, utils.get_item_properties(host, columns))