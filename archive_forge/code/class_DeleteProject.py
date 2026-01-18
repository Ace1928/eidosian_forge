import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteProject(command.Command):
    _description = _('Delete project(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteProject, self).get_parser(prog_name)
        parser.add_argument('projects', metavar='<project>', nargs='+', help=_('Project(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for project in parsed_args.projects:
            try:
                project_obj = utils.find_resource(identity_client.tenants, project)
                identity_client.tenants.delete(project_obj.id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete project with name or ID '%(project)s': %(e)s"), {'project': project, 'e': e})
        if errors > 0:
            total = len(parsed_args.projects)
            msg = _('%(errors)s of %(total)s projects failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)