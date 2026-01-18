import functools
import logging
from cliff import columns as cliff_columns
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListUser(command.Lister):
    _description = _('List users')

    def get_parser(self, prog_name):
        parser = super(ListUser, self).get_parser(prog_name)
        parser.add_argument('--project', metavar='<project>', help=_('Filter users by project (name or ID)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        formatters = {}
        project = None
        if parsed_args.project:
            project = utils.find_resource(identity_client.tenants, parsed_args.project)
            project = project.id
        if parsed_args.long:
            columns = ('ID', 'Name', 'tenantId', 'Email', 'Enabled')
            column_headers = ('ID', 'Name', 'Project', 'Email', 'Enabled')
            project_cache = {}
            try:
                for p in identity_client.tenants.list():
                    project_cache[p.id] = p
            except Exception:
                pass
            formatters['tenantId'] = functools.partial(ProjectColumn, project_cache=project_cache)
        else:
            columns = column_headers = ('ID', 'Name')
        data = identity_client.users.list(tenant_id=project)
        if parsed_args.project:
            d = {}
            for s in data:
                d[s.id] = s
            data = d.values()
        if parsed_args.long:
            for d in data:
                if 'tenant_id' in d._info:
                    d._info['tenantId'] = d._info.pop('tenant_id')
                    d._add_details(d._info)
        return (column_headers, (utils.get_item_properties(s, columns, mixed_case_fields=('tenantId',), formatters=formatters) for s in data))