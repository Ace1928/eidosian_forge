import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class SetRecordSetCommand(command.ShowOne):
    """Set recordset properties"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone_id', help='Zone ID')
        parser.add_argument('id', help='RecordSet ID')
        req_group = parser.add_mutually_exclusive_group()
        req_group.add_argument('--record', help='RecordSet Record, repeat if necessary', action='append')
        description_group = parser.add_mutually_exclusive_group()
        description_group.add_argument('--description', help='Description')
        description_group.add_argument('--no-description', action='store_true')
        ttl_group = parser.add_mutually_exclusive_group()
        ttl_group.add_argument('--ttl', type=int, help='TTL')
        ttl_group.add_argument('--no-ttl', action='store_true')
        common.add_all_common_options(parser)
        common.add_edit_managed_option(parser)
        return parser

    def take_action(self, parsed_args):
        data = {}
        if parsed_args.no_description:
            data['description'] = None
        elif parsed_args.description:
            data['description'] = parsed_args.description
        if parsed_args.no_ttl:
            data['ttl'] = None
        elif parsed_args.ttl:
            data['ttl'] = parsed_args.ttl
        if parsed_args.record:
            data['records'] = parsed_args.record
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        updated = client.recordsets.update(parsed_args.zone_id, parsed_args.id, data)
        _format_recordset(updated)
        return zip(*sorted(updated.items()))