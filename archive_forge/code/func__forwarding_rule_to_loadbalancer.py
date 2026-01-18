from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.compute.drivers.gce import GCEConnection, GCENodeDriver
def _forwarding_rule_to_loadbalancer(self, forwarding_rule):
    """
        Return a Load Balancer object based on a GCEForwardingRule object.

        :param  forwarding_rule: ForwardingRule object
        :type   forwarding_rule: :class:`GCEForwardingRule`

        :return:  LoadBalancer object
        :rtype:   :class:`LoadBalancer`
        """
    extra = {}
    extra['forwarding_rule'] = forwarding_rule
    extra['targetpool'] = forwarding_rule.targetpool
    extra['healthchecks'] = forwarding_rule.targetpool.healthchecks
    return LoadBalancer(id=forwarding_rule.id, name=forwarding_rule.name, state=None, ip=forwarding_rule.address, port=forwarding_rule.extra['portRange'], driver=self, extra=extra)