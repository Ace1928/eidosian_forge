import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
class VRRPV3StateInitialize(VRRPState):

    def master_down(self, ev):
        self.vrrp_router.logger.debug('%s master_down', self.__class__.__name__)

    def adver(self, ev):
        self.vrrp_router.logger.debug('%s adver', self.__class__.__name__)

    def preempt_delay(self, ev):
        self.vrrp_router.logger.warning('%s preempt_delay', self.__class__.__name__)

    def vrrp_received(self, ev):
        self.vrrp_router.logger.debug('%s vrrp_received', self.__class__.__name__)

    def vrrp_shutdown_request(self, ev):
        self.vrrp_router.logger.debug('%s vrrp_shutdown_request', self.__class__.__name__)

    def vrrp_config_change_request(self, ev):
        self.vrrp_router.logger.warning('%s vrrp_config_change_request', self.__class__.__name__)