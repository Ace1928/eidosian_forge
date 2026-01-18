import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _assert_system_nand_project(self):
    if flask.request.args.get('scope.project.id') and flask.request.args.get('scope.system'):
        msg = _('Specify system or project, not both')
        raise exception.ValidationError(msg)