from __future__ import absolute_import, division, print_function
import copy
import datetime
import traceback
import math
import re
from ansible.module_utils.basic import (
from ansible.module_utils.six import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
class ApplicationsParameters(BaseParameters):
    api_map = {'protectionMode': 'protection_mode', 'transactionsPerSecond': 'transactions_per_second', 'newConnections': 'new_connections', 'responseTime': 'response_time', 'activeAlerts': 'active_alerts', 'badTraffic': 'bad_traffic', 'enhancedAnalytics': 'enhanced_analytics', 'badTrafficGrowth': 'bad_traffic_growth'}
    returnables = ['protection_mode', 'id', 'name', 'status', 'transactions_per_second', 'connections', 'new_connections', 'response_time', 'health', 'active_alerts', 'bad_traffic', 'enhanced_analytics', 'bad_traffic_growth']

    @property
    def enhanced_analytics(self):
        return flatten_boolean(self._values['enhanced_analytics'])

    @property
    def bad_traffic_growth(self):
        return flatten_boolean(self._values['bad_traffic_growth'])