from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
class ALBRule:
    """
    AWS ALB listener rule class
    http://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-listeners.html#listener-rules
    """

    def __init__(self, rule_id, is_default, priority, target_group, driver, conditions={}, listener=None):
        self.id = rule_id
        self.is_default = is_default
        self.priority = priority
        self.conditions = conditions
        self._listener = listener
        self._listener_arn = listener.id if listener else None
        self._target_group = target_group
        self._target_group_arn = target_group.id if target_group else None
        self._driver = driver

    @property
    def target_group(self):
        if not self._target_group and self._target_group_arn:
            self._target_group = self._driver.ex_get_target_group(self._target_group_arn)
        return self._target_group

    @target_group.setter
    def target_group(self, val):
        self._target_group = val
        self._target_group_arn = val.id

    @property
    def listener(self):
        if not self._listener and self._listener_arn:
            self._listener = self.driver.ex_get_listener(self._listener_arn)
        return self._listener

    @listener.setter
    def listener(self, val):
        self._listener = val
        self._listener_arn = val.id