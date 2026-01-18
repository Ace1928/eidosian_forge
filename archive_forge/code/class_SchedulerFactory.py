import logging
import threading
from enum import Enum
class SchedulerFactory:

    @classmethod
    def get_scheduler(cls, frozen_vms_resource_pool, policy_name=Policy.ROUNDROBIN):
        if policy_name == Policy.ROUNDROBIN:
            return RoundRobinScheduler(frozen_vms_resource_pool)
        raise RuntimeError(f'Unsupported schedule policy: {policy_name}')