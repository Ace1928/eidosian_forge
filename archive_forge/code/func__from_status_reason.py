import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
def _from_status_reason(self, status_reason):
    """Split the status_reason up into parts.

        Given the following status_reason:
        "Resource DELETE failed: Exception : resources.AResource: foo"

        we are going to return:
        ("Exception", "resources.AResource", "foo")
        """
    parsed = [sp.strip() for sp in status_reason.split(':')]
    if len(parsed) >= 4:
        error = parsed[1]
        message = ': '.join(parsed[3:])
        path = parsed[2].split('.')
    else:
        error = ''
        message = status_reason
        path = []
    return (error, message, path)