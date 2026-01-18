from keystone.common import cache
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import revoke_model
from keystone import notifications
def _assert_not_domain_and_project_scoped(self, domain_id=None, project_id=None):
    if domain_id is not None and project_id is not None:
        msg = _('The revoke call must not have both domain_id and project_id. This is a bug in the Keystone server. The current request is aborted.')
        raise exception.UnexpectedError(exception=msg)