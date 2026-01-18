import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourceUnknownStatus(HeatException):
    msg_fmt = _('%(result)s - Unknown status %(resource_status)s due to "%(status_reason)s"')

    def __init__(self, result=_('Resource failed'), status_reason=_('Unknown'), **kwargs):
        super(ResourceUnknownStatus, self).__init__(result=result, status_reason=status_reason, **kwargs)