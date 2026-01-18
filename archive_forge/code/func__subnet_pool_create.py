import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def _subnet_pool_create(self, cmd, name, is_type_ipv4=True):
    """Make a random subnet pool

        :param string cmd:
            The options for a subnet pool create command, not including
            --pool-prefix and <name>
        :param string name:
            The name of the subnet pool
        :param bool is_type_ipv4:
            Creates an IPv4 pool if True, creates an IPv6 pool otherwise

        Try random subnet ranges because we can not determine ahead of time
        what subnets are already in use, possibly by another test running in
        parallel, try 4 times before failing.
        """
    for i in range(4):
        if is_type_ipv4:
            pool_prefix = '.'.join(map(str, (random.randint(0, 223) for _ in range(2)))) + '.0.0/16'
        else:
            pool_prefix = ':'.join(map(str, (hex(random.randint(0, 65535))[2:] for _ in range(6)))) + ':0:0/96'
        try:
            cmd_output = self.openstack('subnet pool create ' + cmd + ' ' + '--pool-prefix ' + pool_prefix + ' ' + name, parse_output=True)
        except Exception:
            if i == 3:
                raise
            pass
        else:
            break
    return (cmd_output, pool_prefix)