import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
class ShareGroupTypeAccessDeny(command.Command):
    """Deny a project to access a share group type."""
    _description = _('Deny a project to access a share group type (Admin only).')

    def get_parser(self, prog_name):
        parser = super(ShareGroupTypeAccessDeny, self).get_parser(prog_name)
        parser.add_argument('share_group_type', metavar='<share-group-type>', help=_('Share group type name or ID to deny access from'))
        parser.add_argument('projects', metavar='<project>', nargs='+', help=_('Project Name(s) or ID(s) to deny share group type access for.'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        identity_client = self.app.client_manager.identity
        result = 0
        share_group_type = apiutils.find_resource(share_client.share_group_types, parsed_args.share_group_type)
        for project in parsed_args.projects:
            try:
                project_obj = identity_common.find_project(identity_client, project, parsed_args.project_domain)
                share_client.share_group_type_access.remove_project_access(share_group_type, project_obj.id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to deny access for project '%(project)s' to share group type with name or ID '%(share_group_type)s': %(e)s"), {'project': project, 'share_group_type': share_group_type, 'e': e})
        if result > 0:
            total = len(parsed_args.projects)
            msg = _('Failed to deny access to %(result)s of %(total)s projects') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)