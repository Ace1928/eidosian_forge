from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __attach_watchdog(self, entity):
    watchdogs_service = self._service.service(entity.id).watchdogs_service()
    watchdog = self.param('watchdog')
    if watchdog is not None:
        current_watchdog = next(iter(watchdogs_service.list()), None)
        if watchdog.get('model') is None and current_watchdog:
            watchdogs_service.watchdog_service(current_watchdog.id).remove()
            return True
        elif watchdog.get('model') is not None and current_watchdog is None:
            watchdogs_service.add(otypes.Watchdog(model=otypes.WatchdogModel(watchdog.get('model').lower()), action=otypes.WatchdogAction(watchdog.get('action'))))
            return True
        elif current_watchdog is not None:
            if str(current_watchdog.model).lower() != watchdog.get('model').lower() or str(current_watchdog.action).lower() != watchdog.get('action').lower():
                watchdogs_service.watchdog_service(current_watchdog.id).update(otypes.Watchdog(model=otypes.WatchdogModel(watchdog.get('model')), action=otypes.WatchdogAction(watchdog.get('action'))))
                return True
    return False