from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatInvalidActionError(HeatAPIException):
    """The action or operation requested is invalid."""
    code = 400
    title = 'InvalidAction'
    explanation = _('The action or operation requested is invalid')