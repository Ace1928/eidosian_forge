import abc
import datetime as dt
import textwrap
from osc_lib.command import command
class MistralExecutionLister(MistralLister, metaclass=abc.ABCMeta):

    def get_parser(self, parsed_args):
        parser = super(MistralExecutionLister, self).get_parser(parsed_args)
        parser.set_defaults(limit=DEFAULT_LIMIT)
        parser.add_argument('--oldest', help='Display the executions starting from the oldest entries instead of the newest', default=False, action='store_true')
        return parser

    def take_action(self, parsed_args):
        self._validate_parsed_args(parsed_args)
        f = self._get_format_function()
        reverse_results = False
        if parsed_args.marker == '' and parsed_args.sort_dirs == 'asc' and (parsed_args.sort_keys == 'created_at') and (not parsed_args.oldest):
            reverse_results = True
            parsed_args.sort_dirs = 'desc'
        ret = self._get_resources(parsed_args)
        if not isinstance(ret, list):
            ret = [ret]
        if reverse_results:
            ret.reverse()
        data = [f(r)[1] for r in ret]
        if data:
            return (f()[0], data)
        else:
            return f()