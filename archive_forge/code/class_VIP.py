import typing as tp
class VIP(BaseDataModel):

    def __init__(self, vip_address=Unset, vip_network_id=Unset, vip_port_id=Unset, vip_subnet_id=Unset, vip_qos_policy_id=Unset):
        self.vip_address = vip_address
        self.vip_network_id = vip_network_id
        self.vip_port_id = vip_port_id
        self.vip_subnet_id = vip_subnet_id
        self.vip_qos_policy_id = vip_qos_policy_id