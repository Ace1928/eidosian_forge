import functools
import flask
from oslo_log import log
from oslo_policy import opts
from oslo_policy import policy as common_policy
from oslo_utils import strutils
from keystone.common import authorization
from keystone.common import context
from keystone.common import policies
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@staticmethod
def _extract_filter_values(filters):
    """Extract filter data from query params for RBAC enforcement."""
    filters = filters or []
    target = {i: flask.request.args[i] for i in filters if i in flask.request.args}
    if target:
        if LOG.logger.getEffectiveLevel() <= log.DEBUG:
            LOG.debug('RBAC: Adding query filter params (%s)', ', '.join(['%s=%s' % (k, v) for k, v in target.items()]))
    return target