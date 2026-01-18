from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatInvalidParameterValueError(HeatAPIException):
    """A bad or out-of-range value was supplied for the input parameter."""
    code = 400
    title = 'InvalidParameterValue'
    explanation = _('A bad or out-of-range value was supplied')