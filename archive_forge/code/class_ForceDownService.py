from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
class ForceDownService(command.ShowOne):
    """Force the Zun service to down or up."""
    log = logging.getLogger(__name__ + '.ForceDownService')

    def get_parser(self, prog_name):
        parser = super(ForceDownService, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help='Name of host')
        parser.add_argument('binary', metavar='<binary>', help='Name of the binary to disable')
        parser.add_argument('--unset', dest='force_down', help='Unset the force state down of service', action='store_false', default=True)
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        host = parsed_args.host
        binary = parsed_args.binary
        force_down = parsed_args.force_down
        res = client.services.force_down(host, binary, force_down)
        columns = ('Host', 'Binary', 'Forced_down')
        return (columns, utils.get_dict_properties(res[1]['service'], columns))