import abc
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import api as vrrp_api
class VRRPState(object, metaclass=abc.ABCMeta):

    def __init__(self, vrrp_router):
        super(VRRPState, self).__init__()
        self.vrrp_router = vrrp_router

    @abc.abstractmethod
    def master_down(self, ev):
        pass

    @abc.abstractmethod
    def adver(self, ev):
        pass

    @abc.abstractmethod
    def preempt_delay(self, ev):
        pass

    @abc.abstractmethod
    def vrrp_received(self, ev):
        pass

    @abc.abstractmethod
    def vrrp_shutdown_request(self, ev):
        pass

    @abc.abstractmethod
    def vrrp_config_change_request(self, ev):
        pass