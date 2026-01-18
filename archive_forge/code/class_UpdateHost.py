import logging
from blazarclient import command
from blazarclient import exception
class UpdateHost(command.UpdateCommand):
    """Update attributes of a host."""
    resource = 'host'
    json_indent = 4
    log = logging.getLogger(__name__ + '.UpdateHost')
    name_key = 'hypervisor_hostname'
    id_pattern = HOST_ID_PATTERN

    def get_parser(self, prog_name):
        parser = super(UpdateHost, self).get_parser(prog_name)
        parser.add_argument('--extra', metavar='<key>=<value>', action='append', dest='extra_capabilities', default=[], help='Extra capabilities key/value pairs to update for the host')
        return parser

    def args2body(self, parsed_args):
        params = {}
        extras = {}
        if parsed_args.extra_capabilities:
            for capa in parsed_args.extra_capabilities:
                key, _sep, value = capa.partition('=')
                extras[key] = value
            params['values'] = extras
        return params