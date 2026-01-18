from cliff import lister
from oslo_serialization import jsonutils
from aodhclient import utils
class CliAlarmHistorySearch(lister.Lister):
    """Show history for all alarms based on query"""
    COLS = ('alarm_id', 'timestamp', 'type', 'detail')

    def get_parser(self, prog_name):
        parser = super(CliAlarmHistorySearch, self).get_parser(prog_name)
        (parser.add_argument('--query', help='Rich query supported by aodh, e.g. project_id!=my-id user_id=foo or user_id=bar'),)
        return parser

    def take_action(self, parsed_args):
        query = None
        if parsed_args.query:
            query = jsonutils.dumps(utils.search_query_builder(parsed_args.query))
        history = utils.get_client(self).alarm_history.search(query=query)
        return utils.list2cols(self.COLS, history)