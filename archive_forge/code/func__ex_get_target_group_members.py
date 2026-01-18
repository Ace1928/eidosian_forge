from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def _ex_get_target_group_members(self, target_group):
    """
        Return a list of target group member dicts.

        :param target_group: target group to fetch members for
        :type target_group: :class:`ALBTargetGroup`

        :return: list of target group members
        :rtype: ``list`` of :class:`Member`
        """
    params = {'Action': 'DescribeTargetHealth', 'TargetGroupArn': target_group.id}
    data = self.connection.request(ROOT, params=params).object
    target_group_members = []
    for tg_member in self._to_target_group_members(data):
        tg_member.extra['target_group'] = target_group
        target_group_members.append(tg_member)
    return target_group_members