import socket
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.api.aws import exception
from heat.api.aws import utils as api_utils
from heat.common import exception as heat_exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
def _get_template(self, req):
    """Get template file contents, either from local file or URL."""
    if 'TemplateBody' in req.params:
        LOG.debug('TemplateBody ...')
        return req.params['TemplateBody']
    elif 'TemplateUrl' in req.params:
        url = req.params['TemplateUrl']
        LOG.debug('TemplateUrl %s' % url)
        try:
            return urlfetch.get(url)
        except IOError as exc:
            msg = _('Failed to fetch template: %s') % exc
            raise exception.HeatInvalidParameterValueError(detail=msg)
    return None