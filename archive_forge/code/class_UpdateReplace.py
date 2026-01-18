import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class UpdateReplace(Exception):
    """Raised when resource update requires replacement."""

    def __init__(self, resource_name='Unknown'):
        msg = _('The Resource %s requires replacement.') % resource_name
        super(Exception, self).__init__(str(msg))