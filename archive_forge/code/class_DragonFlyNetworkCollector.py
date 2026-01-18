from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
class DragonFlyNetworkCollector(NetworkCollector):
    _fact_class = DragonFlyNetwork
    _platform = 'DragonFly'