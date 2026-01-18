from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatActionInProgressError(HeatAPIException):
    """Cannot perform action on stack in its current state."""
    code = 400
    title = 'InvalidAction'
    explanation = 'Cannot perform action on stack while other actions are ' + 'in progress'