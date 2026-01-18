from troveclient import base
from troveclient import common
class SecurityGroupRules(base.ManagerWithFind):
    """Manage :class:`SecurityGroupRules` resources."""
    resource_class = SecurityGroupRule

    def create(self, group_id, cidr):
        """Create a new security group rule."""
        body = {'security_group_rule': {'group_id': group_id, 'cidr': cidr}}
        return self._create('/security-group-rules', body, 'security_group_rule', return_raw=True)

    def delete(self, security_group_rule):
        """Delete the specified security group rule.

        :param security_group_rule: The security group rule to delete
        """
        url = '/security-group-rules/%s' % base.getid(security_group_rule)
        resp, body = self.api.client.delete(url)
        common.check_for_exceptions(resp, body, url)

    def list(self):
        pass