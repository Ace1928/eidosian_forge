import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _list_role_assignments_for_tree(self):
    filters = ['group.id', 'role.id', 'scope.domain.id', 'scope.project.id', 'scope.OS-INHERIT:inherited_to', 'user.id']
    project_id = flask.request.args.get('scope.project.id')
    target = None
    if project_id:
        target = {'project': PROVIDERS.resource_api.get_project(project_id)}
        target['domain_id'] = target['project']['domain_id']
    ENFORCER.enforce_call(action='identity:list_role_assignments_for_tree', filters=filters, target_attr=target)
    if not project_id:
        msg = _('scope.project.id must be specified if include_subtree is also specified')
        raise exception.ValidationError(message=msg)
    return self._build_role_assignments_list(include_subtree=True)