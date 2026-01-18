from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _ex_get_rules_for_listener(self, listener):
    """
        Get list of rules associated with listener.

        :param listener: Listener object to fetch rules for
        :type listener: :class:`ALBListener`

        :return: List of rules
        :rtype: ``list`` of :class:`ALBListener`
        """
    params = {'Action': 'DescribeRules', 'ListenerArn': listener.id}
    data = self.connection.request(ROOT, params=params).object
    rules = self._to_rules(data)
    for rule in rules:
        rule.listener = listener
    return rules