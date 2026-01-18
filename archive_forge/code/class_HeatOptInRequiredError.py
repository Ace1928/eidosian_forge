from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatOptInRequiredError(HeatAPIException):
    """The AWS Access Key ID needs a subscription for the service."""
    code = 403
    title = 'OptInRequired'
    explanation = _('The AWS Access Key ID needs a subscription for the service')