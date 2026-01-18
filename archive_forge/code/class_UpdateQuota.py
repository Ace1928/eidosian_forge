from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.i18n import _
class UpdateQuota(command.ShowOne):
    """Update the quotas of the project"""
    log = logging.getLogger(__name__ + '.UpdateQuota')

    def get_parser(self, prog_name):
        parser = super(UpdateQuota, self).get_parser(prog_name)
        parser.add_argument('--containers', metavar='<containers>', help='The number of containers allowed per project')
        parser.add_argument('--memory', metavar='<memory>', help='The number of megabytes of container RAM allowed per project')
        parser.add_argument('--cpu', metavar='<cpu>', help='The number of container cores or vCPUs allowed per project')
        parser.add_argument('--disk', metavar='<disk>', help='The number of gigabytes of container Disk allowed per project')
        parser.add_argument('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['containers'] = parsed_args.containers
        opts['memory'] = parsed_args.memory
        opts['cpu'] = parsed_args.cpu
        opts['disk'] = parsed_args.disk
        quota = client.quotas.update(parsed_args.project_id, **opts)
        columns = _quota_columns(quota)
        return (columns, utils.get_item_properties(quota, columns))