import argparse
import datetime
import logging
import re
from oslo_serialization import jsonutils
from oslo_utils import strutils
from blazarclient import command
from blazarclient import exception
class CreateLeaseBase(command.CreateCommand):
    """Create a lease."""
    resource = 'lease'
    json_indent = 4
    log = logging.getLogger(__name__ + '.CreateLease')
    default_start = 'now'
    default_end = _utc_now() + datetime.timedelta(days=1)

    def get_parser(self, prog_name):
        parser = super(CreateLeaseBase, self).get_parser(prog_name)
        parser.add_argument('name', metavar=self.resource.upper(), help='Name for the %s' % self.resource)
        parser.add_argument('--start-date', dest='start', help='Time (YYYY-MM-DD HH:MM) UTC TZ for starting the lease (default: current time on the server)', default=self.default_start)
        parser.add_argument('--end-date', dest='end', help='Time (YYYY-MM-DD HH:MM) UTC TZ for ending the lease (default: 24h from now)', default=self.default_end)
        parser.add_argument('--before-end-date', dest='before_end', help='Time (YYYY-MM-DD HH:MM) UTC TZ for taking an action before the end of the lease (default: depends on system default)', default=None)
        parser.add_argument('--reservation', metavar='<key=value>', action='append', dest='reservations', help='key/value pairs for creating a generic reservation. Specify option multiple times to create multiple reservations. ', default=[])
        parser.add_argument('--event', metavar='<event_type=str,event_date=time>', action='append', dest='events', help='Creates an event with key/value pairs for the lease. Specify option multiple times to create multiple events. event_type: type of event (e.g. notification). event_date: Time for event (YYYY-MM-DD HH:MM) UTC TZ. ', default=[])
        return parser

    def args2body(self, parsed_args):
        params = self._generate_params(parsed_args)
        if not params['reservations']:
            raise exception.IncorrectLease
        return params

    def _generate_params(self, parsed_args):
        params = {}
        if parsed_args.name:
            params['name'] = parsed_args.name
        if not isinstance(parsed_args.start, datetime.datetime):
            if parsed_args.start != 'now':
                try:
                    parsed_args.start = datetime.datetime.strptime(parsed_args.start, '%Y-%m-%d %H:%M')
                except ValueError:
                    raise exception.IncorrectLease
        if not isinstance(parsed_args.end, datetime.datetime):
            try:
                parsed_args.end = datetime.datetime.strptime(parsed_args.end, '%Y-%m-%d %H:%M')
            except ValueError:
                raise exception.IncorrectLease
        if parsed_args.start == 'now':
            start = _utc_now()
        else:
            start = parsed_args.start
        if start > parsed_args.end:
            raise exception.IncorrectLease
        if parsed_args.before_end:
            try:
                parsed_args.before_end = datetime.datetime.strptime(parsed_args.before_end, '%Y-%m-%d %H:%M')
            except ValueError:
                raise exception.IncorrectLease
            if parsed_args.before_end < start or parsed_args.end < parsed_args.before_end:
                raise exception.IncorrectLease
            params['before_end'] = datetime.datetime.strftime(parsed_args.before_end, '%Y-%m-%d %H:%M')
        if parsed_args.start == 'now':
            params['start'] = parsed_args.start
        else:
            params['start'] = datetime.datetime.strftime(parsed_args.start, '%Y-%m-%d %H:%M')
        params['end'] = datetime.datetime.strftime(parsed_args.end, '%Y-%m-%d %H:%M')
        params['reservations'] = []
        params['events'] = []
        reservations = []
        for res_str in parsed_args.reservations:
            err_msg = "Invalid reservation argument '%s'. Reservation arguments must be of the form --reservation <key=value>" % res_str
            if 'physical:host' in res_str:
                defaults = CREATE_RESERVATION_KEYS['physical:host']
            elif 'virtual:instance' in res_str:
                defaults = CREATE_RESERVATION_KEYS['virtual:instance']
            elif 'virtual:floatingip' in res_str:
                defaults = CREATE_RESERVATION_KEYS['virtual:floatingip']
            else:
                defaults = CREATE_RESERVATION_KEYS['others']
            res_info = self._parse_params(res_str, defaults, err_msg)
            reservations.append(res_info)
        if reservations:
            params['reservations'] += reservations
        events = []
        for event_str in parsed_args.events:
            err_msg = "Invalid event argument '%s'. Event arguments must be of the form --event <event_type=str,event_date=time>" % event_str
            event_info = {'event_type': '', 'event_date': ''}
            for kv_str in event_str.split(','):
                try:
                    k, v = kv_str.split('=', 1)
                except ValueError:
                    raise exception.IncorrectLease(err_msg)
                if k in event_info:
                    event_info[k] = v
                else:
                    raise exception.IncorrectLease(err_msg)
            if not event_info['event_type'] and (not event_info['event_date']):
                raise exception.IncorrectLease(err_msg)
            event_date = event_info['event_date']
            try:
                date = datetime.datetime.strptime(event_date, '%Y-%m-%d %H:%M')
                event_date = datetime.datetime.strftime(date, '%Y-%m-%d %H:%M')
                event_info['event_date'] = event_date
            except ValueError:
                raise exception.IncorrectLease
            events.append(event_info)
        if events:
            params['events'] = events
        return params

    def _parse_params(self, str_params, default, err_msg):
        request_params = {}
        prog = re.compile('^(?:(.*),)?(%s)=(.*)$' % '|'.join(default.keys()))
        while str_params != '':
            match = prog.search(str_params)
            if match is None:
                raise exception.IncorrectLease(err_msg)
            self.log.info('Matches: %s', match.groups())
            k, v = match.group(2, 3)
            if k in request_params.keys():
                raise exception.DuplicatedLeaseParameters(err_msg)
            elif strutils.is_int_like(v):
                request_params[k] = int(v)
            elif isinstance(default[k], list):
                request_params[k] = jsonutils.loads(v)
            else:
                request_params[k] = v
            str_params = match.group(1) if match.group(1) else ''
        request_params.update({k: v for k, v in default.items() if k not in request_params.keys() and v is not None})
        return request_params