import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def _info(self):
    resources_used, resources_total = self._get_resource_usage()
    now = time.time()
    idle_times = [now - t for t in self.last_used_time_by_ip.values()]
    heartbeat_times = [now - t for t in self.last_heartbeat_time_by_ip.values()]
    most_delayed_heartbeats = sorted(self.last_heartbeat_time_by_ip.items(), key=lambda pair: pair[1])[:5]
    most_delayed_heartbeats = {ip: now - t for ip, t in most_delayed_heartbeats}

    def format_resource(key, value):
        if key in ['object_store_memory', 'memory']:
            return '{} GiB'.format(round(value / (1024 * 1024 * 1024), 2))
        else:
            return round(value, 2)
    return {'ResourceUsage': ', '.join(['{}/{} {}'.format(format_resource(rid, resources_used[rid]), format_resource(rid, resources_total[rid]), rid) for rid in sorted(resources_used) if not rid.startswith('node:')]), 'NodeIdleSeconds': 'Min={} Mean={} Max={}'.format(int(min(idle_times)) if idle_times else -1, int(float(sum(idle_times)) / len(idle_times)) if idle_times else -1, int(max(idle_times)) if idle_times else -1), 'TimeSinceLastHeartbeat': 'Min={} Mean={} Max={}'.format(int(min(heartbeat_times)) if heartbeat_times else -1, int(float(sum(heartbeat_times)) / len(heartbeat_times)) if heartbeat_times else -1, int(max(heartbeat_times)) if heartbeat_times else -1), 'MostDelayedHeartbeats': most_delayed_heartbeats}