from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.i18n import _
class DeleteQuota(command.Command):
    """Delete quota of the project"""
    log = logging.getLogger(__name__ + '.DeleteQuota')

    def get_parser(self, prog_name):
        parser = super(DeleteQuota, self).get_parser(prog_name)
        parser.add_argument('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        try:
            client.quotas.delete(parsed_args.project_id)
            print(_('Request to delete quotas has been accepted.'))
        except Exception as e:
            print('Delete for quotas failed: %(e)s' % {'e': e})