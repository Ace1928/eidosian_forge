from urllib import parse
from novaclient import base
class InstanceUsageAuditLogManager(base.Manager):
    resource_class = InstanceUsageAuditLog

    def get(self, before=None):
        """Get server usage audits.

        :param before: Filters the response by the date and time
                       before which to list usage audits.
        """
        if before:
            return self._get('/os-instance_usage_audit_log/%s' % parse.quote(before, safe=''), 'instance_usage_audit_log')
        else:
            return self._get('/os-instance_usage_audit_log', 'instance_usage_audit_logs')