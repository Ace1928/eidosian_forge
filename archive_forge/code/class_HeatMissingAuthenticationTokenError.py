from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatMissingAuthenticationTokenError(HeatAPIException):
    """Does not contain a valid AWS Access Key or certificate.

    Request must contain either a valid (registered) AWS Access Key ID
    or X.509 certificate.
    """
    code = 403
    title = 'MissingAuthenticationToken'
    explanation = _('Does not contain a valid AWS Access Key or certificate')