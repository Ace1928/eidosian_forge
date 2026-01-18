from .default import DefaultDeviceHandler
from ncclient.operations.third_party.iosxe.rpc import SaveConfig
from ncclient.xml_ import BASE_NS_1_0
import logging
def add_additional_ssh_connect_params(self, kwargs):
    kwargs['unknown_host_cb'] = iosxe_unknown_host_cb