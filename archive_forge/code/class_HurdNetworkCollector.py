from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.facts.network.base import Network, NetworkCollector
class HurdNetworkCollector(NetworkCollector):
    _platform = 'GNU'
    _fact_class = HurdPfinetNetwork