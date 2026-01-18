from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
class PoweroffMachine(RPC):
    """*poweroff-machine* RPC (flowmon)"""
    DEPENDS = ['urn:liberouter:param:netconf:capability:power-control:1.0']

    def request(self):
        return self._request(new_ele(qualify('poweroff-machine', PC_URN)))