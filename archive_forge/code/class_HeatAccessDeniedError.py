from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatAccessDeniedError(HeatAPIException):
    """Authentication fails due to user IAM group memberships.

    This is the response given when authentication fails due to user
    IAM group memberships meaning we deny access.
    """
    code = 403
    title = 'AccessDenied'
    explanation = _('User is not authorized to perform action')