import fnmatch
import logging
from oslo_config import cfg
from stevedore import dispatch
import yaml
from oslo_messaging.notify import notifier
def _get_drivers_for_message(self, group, event_type, priority):
    """Which drivers should be called for this event_type
           or priority.
        """
    accepted_drivers = set()
    for driver, rules in group.items():
        checks = []
        for key, patterns in rules.items():
            if key == 'accepted_events':
                c = [fnmatch.fnmatch(event_type, p) for p in patterns]
                checks.append(any(c))
            if key == 'accepted_priorities':
                c = [fnmatch.fnmatch(priority, p.lower()) for p in patterns]
                checks.append(any(c))
        if all(checks):
            accepted_drivers.add(driver)
    return list(accepted_drivers)