from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatRequestExpiredError(HeatAPIException):
    """Request expired or more than 15 minutes in the future.

    Request is past expires date or the request date (either with 15 minute
    padding), or the request date occurs more than 15 minutes in the future.
    """
    code = 400
    title = 'RequestExpired'
    explanation = _('Request expired or more than 15mins in the future')