from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
@classmethod
def _validate_quota(cls, quota_property, quota_size, total_size):
    err_message = _('Invalid quota %(property)s value(s): %(value)s. Can not be less than the current usage value(s): %(total)s.')
    if quota_size < total_size:
        message_format = {'property': quota_property, 'value': quota_size, 'total': total_size}
        raise ValueError(err_message % message_format)