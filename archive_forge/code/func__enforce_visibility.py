from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def _enforce_visibility(self, visibility):
    try:
        policy._enforce_image_visibility(self.enforcer, self._context, visibility, self._target)
    except exception.Forbidden as e:
        raise webob.exc.HTTPForbidden(explanation=str(e))