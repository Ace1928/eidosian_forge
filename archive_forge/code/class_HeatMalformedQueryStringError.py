from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatMalformedQueryStringError(HeatAPIException):
    """The query string is malformed."""
    code = 404
    title = 'MalformedQueryString'
    explanation = _('The query string is malformed')