import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
class CliAlarmList(lister.Lister):
    """List alarms"""

    @staticmethod
    def split_filter_param(param):
        key, eq_op, value = param.partition('=')
        if not eq_op:
            msg = 'Malformed parameter(%s). Use the key=value format.' % param
            raise ValueError(msg)
        return (key, value)

    def get_parser(self, prog_name):
        parser = super(CliAlarmList, self).get_parser(prog_name)
        exclusive_group = parser.add_mutually_exclusive_group()
        exclusive_group.add_argument('--query', help='Rich query supported by aodh, e.g. project_id!=my-id user_id=foo or user_id=bar')
        exclusive_group.add_argument('--filter', dest='filter', metavar='<KEY1=VALUE1;KEY2=VALUE2...>', type=self.split_filter_param, action='append', help='Filter parameters to apply on returned alarms.')
        parser.add_argument('--limit', type=int, metavar='<LIMIT>', help='Number of resources to return (Default is server default)')
        parser.add_argument('--marker', metavar='<MARKER>', help='Last item of the previous listing. Return the next results after this value,the supported marker is alarm_id.')
        parser.add_argument('--sort', action='append', metavar='<SORT_KEY:SORT_DIR>', help='Sort of resource attribute, e.g. name:asc')
        return parser

    def take_action(self, parsed_args):
        if parsed_args.query:
            if any([parsed_args.limit, parsed_args.sort, parsed_args.marker]):
                raise exceptions.CommandError('Query and pagination options are mutually exclusive.')
            query = jsonutils.dumps(utils.search_query_builder(parsed_args.query))
            alarms = utils.get_client(self).alarm.query(query=query)
        else:
            filters = dict(parsed_args.filter) if parsed_args.filter else None
            alarms = utils.get_client(self).alarm.list(filters=filters, sorts=parsed_args.sort, limit=parsed_args.limit, marker=parsed_args.marker)
        return utils.list2cols(ALARM_LIST_COLS, alarms)