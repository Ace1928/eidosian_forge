import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
@VRRPRouter.register(vrrp.VRRP_VERSION_V3)
class VRRPRouterV3(VRRPRouter):
    _STATE_MAP = {vrrp_event.VRRP_STATE_INITIALIZE: VRRPV3StateInitialize, vrrp_event.VRRP_STATE_MASTER: VRRPV3StateMaster, vrrp_event.VRRP_STATE_BACKUP: VRRPV3StateBackup}

    def __init__(self, *args, **kwargs):
        super(VRRPRouterV3, self).__init__(*args, **kwargs)

    def start(self):
        self.state_change(vrrp_event.VRRP_STATE_INITIALIZE)
        if self.config.address_owner or self.config.admin_state == 'master':
            self.send_advertisement()
            self.state_change(vrrp_event.VRRP_STATE_MASTER)
            self.adver_timer.start(self.config.advertisement_interval)
        else:
            params = self.params
            params.master_adver_interval = self.config.advertisement_interval
            self.state_change(vrrp_event.VRRP_STATE_BACKUP)
            self.master_down_timer.start(params.master_down_interval)
        self.stats_out_timer.start(self.statistics.statistics_interval)
        super(VRRPRouterV3, self).start()