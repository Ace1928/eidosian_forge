from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.i18n import _
class GetQuota(command.ShowOne):
    """Get quota of the project"""
    log = logging.getLogger(__name__ + '.GetQuota')

    def get_parser(self, prog_name):
        parser = super(GetQuota, self).get_parser(prog_name)
        parser.add_argument('--usages', action='store_true', help='Whether show quota usage statistic or not')
        parser.add_argument('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        quota = client.quotas.get(parsed_args.project_id, usages=parsed_args.usages)
        columns = _quota_columns(quota)
        return (columns, utils.get_item_properties(quota, columns))