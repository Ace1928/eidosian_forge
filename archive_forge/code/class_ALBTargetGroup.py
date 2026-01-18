from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
class ALBTargetGroup:
    """
    AWS ALB target group class
    http://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html
    """

    def __init__(self, target_group_id, name, protocol, port, vpc, driver, health_check_timeout=5, health_check_port='traffic-port', health_check_path='/', health_check_proto='HTTP', health_check_matcher='200', health_check_interval=30, healthy_threshold=5, unhealthy_threshold=2, balancers=[], members=[]):
        self.id = target_group_id
        self.name = name
        self.protocol = protocol
        self.port = port
        self.vpc = vpc
        self.health_check_timeout = health_check_timeout
        self.health_check_port = health_check_port
        self.health_check_path = health_check_path
        self.health_check_proto = health_check_proto
        self.health_check_matcher = health_check_matcher
        self.health_check_interval = health_check_interval
        self.healthy_threshold = healthy_threshold
        self.unhealthy_threshold = unhealthy_threshold
        self._balancers = balancers
        self._balancers_arns = [lb.id for lb in balancers] if balancers else []
        self._members = members
        self._members_ids = [mb.id for mb in members] if members else []
        self._driver = driver

    @property
    def balancers(self):
        if not self._balancers and self._balancers_arns:
            self._balancers = []
            for balancer_arn in self._balancers_arns:
                self._balancers.append(self._driver.get_balancer(balancer_arn))
        return self._balancers

    @balancers.setter
    def balancers(self, val):
        self._balancers = val
        self._balancers_arns = [lb.id for lb in val] if val else []

    @property
    def members(self):
        if not self._members:
            mbrs = self._driver._ex_get_target_group_members(self)
            self._members = mbrs
            self._members_ids = [mb.id for mb in mbrs] if mbrs else []
        return self._members

    @members.setter
    def members(self, val):
        self._members = val
        self._members_ids = [mb.id for mb in val] if val else []