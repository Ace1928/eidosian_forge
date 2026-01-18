from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ShareTypeAccessAllow(command.Command):
    """Add access for share type."""
    _description = _('Add access for share type')

    def get_parser(self, prog_name):
        parser = super(ShareTypeAccessAllow, self).get_parser(prog_name)
        parser.add_argument('share_type', metavar='<share_type>', help=_('Share type name or ID to add access to'))
        parser.add_argument('project_id', metavar='<project_id>', help=_('Project ID to add share type access for'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type)
        try:
            share_client.share_type_access.add_project_access(share_type, parsed_args.project_id)
        except Exception as e:
            raise exceptions.CommandError('Failed to add access to share type : %s' % e)