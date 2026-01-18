from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatInternalFailureError(HeatAPIException):
    """The request processing has failed due to some unknown error."""
    code = 500
    title = 'InternalFailure'
    explanation = _('The request processing has failed due to an internal error')
    err_type = 'Server'