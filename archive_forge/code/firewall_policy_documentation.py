from openstack.exceptions import HttpException
from openstack import resource
from openstack import utils
Remove a firewall_rule from a firewall_policy.

        :param session: The session to communicate through.
        :type session: :class:`~openstack.session.Session`
        :param dict body: The body requested to be updated on the router

        :returns: The updated firewall policy
        :rtype: :class:`~openstack.network.v2.firewall_policy.FirewallPolicy`

        :raises: :class:`~openstack.exceptions.HttpException` on error.
        