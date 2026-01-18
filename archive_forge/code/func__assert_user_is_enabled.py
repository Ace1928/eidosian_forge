from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from pycadf import reason
from pycadf import resource
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _assert_user_is_enabled(self, user_ref):
    try:
        PROVIDERS.identity_api.assert_user_enabled(user_id=user_ref['id'], user=user_ref)
    except AssertionError as e:
        LOG.warning(e)
        raise exception.Unauthorized from e