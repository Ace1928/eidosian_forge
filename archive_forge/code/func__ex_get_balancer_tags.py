from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _ex_get_balancer_tags(self, balancer):
    """
        Get a dict of load balancer tags.

        :param balancer: Load balancer to fetch tags for
        :type balancer: :class:`LoadBalancer`

        :return: Dictionary of tags (name/value) for load balancer
        :rtype: ``dict``
        """
    params = {'Action': 'DescribeTags', 'ResourceArns.member.1': balancer.id}
    data = self.connection.request(ROOT, params=params).object
    return self._to_tags(data)