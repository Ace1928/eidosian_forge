from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatInvalidParameterCombinationError(HeatAPIException):
    """Parameters that must not be used together were used together."""
    code = 400
    title = 'InvalidParameterCombination'
    explanation = _('Incompatible parameters were used together')