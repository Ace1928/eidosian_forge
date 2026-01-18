from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatServiceUnavailableError(HeatAPIException):
    """The request has failed due to a temporary failure of the server."""
    code = 503
    title = 'ServiceUnavailable'
    explanation = _('Service temporarily unavailable')
    err_type = 'Server'