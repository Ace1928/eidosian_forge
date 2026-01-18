import time
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import monitor as vrrp_monitor
from os_ken.services.protocols.vrrp import router as vrrp_router
def _proxy_event(self, ev):
    name = ev.instance_name
    instance = self._instances.get(name, None)
    if not instance:
        self.logger.info('unknown vrrp router %s', name)
        return
    self.send_event(instance.name, ev)