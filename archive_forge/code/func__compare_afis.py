from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospf_interfaces import (
def _compare_afis(self, want, have):
    """Leverages the base class `compare()` method and
        populates the list of commands to be run by comparing
        the `want` and `have` data with the `parsers` defined
        for the Ospf_interfaces network resource.
        """
    parsers = ['name', 'process', 'adjacency', 'authentication', 'bfd', 'cost', 'database_filter', 'dead_interval', 'demand_circuit', 'flood_reduction', 'hello_interval', 'lls', 'manet', 'mtu_ignore', 'multi_area', 'neighbor', 'network', 'prefix_suppression', 'priority', 'resync_timeout', 'retransmit_interval', 'shutdown', 'transmit_delay', 'ttl_security']
    for afi in ('ipv4', 'ipv6'):
        wacls = want.pop(afi, {})
        hacls = have.pop(afi, {})
        self.compare(parsers=parsers, want=wacls, have=hacls)