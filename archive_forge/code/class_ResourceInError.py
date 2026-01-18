import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourceInError(HeatException):
    msg_fmt = _('Went to status %(resource_status)s due to "%(status_reason)s"')

    def __init__(self, status_reason=_('Unknown'), **kwargs):
        super(ResourceInError, self).__init__(status_reason=status_reason, **kwargs)