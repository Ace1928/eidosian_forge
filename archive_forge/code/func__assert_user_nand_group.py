import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _assert_user_nand_group(self):
    if flask.request.args.get('user.id') and flask.request.args.get('group.id'):
        msg = _('Specify a user or group, not both')
        raise exception.ValidationError(msg)