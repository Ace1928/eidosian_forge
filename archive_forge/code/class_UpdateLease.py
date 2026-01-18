import argparse
import datetime
import logging
import re
from oslo_serialization import jsonutils
from oslo_utils import strutils
from blazarclient import command
from blazarclient import exception
class UpdateLease(command.UpdateCommand):
    """Update a lease."""
    resource = 'lease'
    json_indent = 4
    log = logging.getLogger(__name__ + '.UpdateLease')

    def get_parser(self, prog_name):
        parser = super(UpdateLease, self).get_parser(prog_name)
        parser.add_argument('--name', help='New name for the lease', default=None)
        parser.add_argument('--reservation', metavar='<key=value>', action='append', help='Reservation values to update. The reservation must be selected with the id=<reservation-id> key-value pair.', default=None)
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--prolong-for', help='Time to prolong lease for', default=None)
        group.add_argument('--prolong_for', help=argparse.SUPPRESS, default=None)
        group.add_argument('--reduce-by', help='Time to reduce lease by', default=None)
        group.add_argument('--end-date', help='end date of the lease', default=None)
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--defer-by', help='Time to defer the lease start', default=None)
        group.add_argument('--advance-by', help='Time to advance the lease start', default=None)
        group.add_argument('--start-date', help='start date of the lease', default=None)
        return parser

    def args2body(self, parsed_args):
        params = {}
        if parsed_args.name:
            params['name'] = parsed_args.name
        if parsed_args.prolong_for:
            params['prolong_for'] = parsed_args.prolong_for
        if parsed_args.reduce_by:
            params['reduce_by'] = parsed_args.reduce_by
        if parsed_args.end_date:
            params['end_date'] = parsed_args.end_date
        if parsed_args.defer_by:
            params['defer_by'] = parsed_args.defer_by
        if parsed_args.advance_by:
            params['advance_by'] = parsed_args.advance_by
        if parsed_args.start_date:
            params['start_date'] = parsed_args.start_date
        if parsed_args.reservation:
            keys = set(['id', 'min', 'max', 'hypervisor_properties', 'resource_properties', 'vcpus', 'memory_mb', 'disk_gb', 'amount', 'affinity', 'amount', 'network_id', 'required_floatingips'])
            list_keys = ['required_floatingips']
            params['reservations'] = []
            reservations = []
            for res_str in parsed_args.reservation:
                err_msg = "Invalid reservation argument '%s'. Reservation arguments must be of the form --reservation <key=value>" % res_str
                res_info = {}
                prog = re.compile('^(?:(.*),)?(%s)=(.*)$' % '|'.join(keys))

                def parse_params(params):
                    match = prog.search(params)
                    if match:
                        k, v = match.group(2, 3)
                        if k in list_keys:
                            v = jsonutils.loads(v)
                        elif strutils.is_int_like(v):
                            v = int(v)
                        res_info[k] = v
                        if match.group(1) is not None:
                            parse_params(match.group(1))
                parse_params(res_str)
                if res_info:
                    if 'id' not in res_info:
                        raise exception.IncorrectLease('The key-value pair id=<reservation_id> is required for the --reservation argument')
                    reservations.append(res_info)
            if not reservations:
                raise exception.IncorrectLease(err_msg)
            params['reservations'] = reservations
        return params